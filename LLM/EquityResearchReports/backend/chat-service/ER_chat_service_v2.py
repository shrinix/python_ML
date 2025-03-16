# __import__('pysqlite3')
import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import logging
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import os
import argparse
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAI
from langchain_openai import OpenAIEmbeddings
from chromadb.config import Settings
import tiktoken
import hashlib
import time
import pandas as pd
import requests

#set environment variable to disable ChromadB telemetry
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

# Set up logging
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(level=logging.INFO)
# Ensure the logs directory exists

logger = logging.getLogger(__name__)
log_filename = os.path.join(log_dir, f"ER-chat-service-{time.strftime('%Y-%m-%d_%H-%M-%S')}.log")
sys.stdout = open(log_filename, 'a', buffering=1)
sys.stderr = open(log_filename, 'a')
log_filename = f"ER-chat-service-{time.strftime('%Y-%m-%d_%H-%M-%S')}.log"
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

print("Loggint setup completed: ", log_dir)

# Set display options to control wrapping
pd.set_option('display.max_columns', None)  # Do not truncate the list of columns
pd.set_option('display.max_rows', None)     # Do not truncate the list of rows
pd.set_option('display.width', 1000)        # Set the display width to a large value
pd.set_option('display.max_colwidth', 50)   # Set the maximum column width
pd.set_option('display.colheader_justify', 'left')  # Justify column headers to the left

#print value of environment variable $VIRTUAL_ENV
print("Virtual environment: ", os.getenv('VIRTUAL_ENV'))
#print value of environment variable $PYTHONPATH
print("Python path: ", os.getenv('PYTHONPATH'))
#print versions of langchain and langchain_openai
# print("Langchain OpenAI version: ", langchain_openai.__version__)
    

#check if OPEN_AI_API_KEY is set
if 'OPENAI_API_KEY' not in os.environ:
    print("Please set the OPENAI_API_KEY environment variable")
    sys.exit(1)

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Parse command line arguments
parser = argparse.ArgumentParser(description='Start the Flask application with a specific .env file.')
parser.add_argument('--env_file', type=str, required=False, help='Path to the .env file')
args = parser.parse_args()

if args.env_file:
    print(f"Loading environment variables from {args.env_file}")
    # Load environment variables from the specified .env file
    load_dotenv(args.env_file)
else:
    print("No .env file specified. Loading environment variables from backend.env")
    # Load environment variables from .env file
    load_dotenv('backend.env')

app = Flask(__name__)
# Get CORS origins from environment variables
cors_origins = os.getenv('CORS_ORIGINS', '*')

# # Enable CORS for specific IP address
CORS(app,supports_credentials=True, resources={r"/*": {"origins": cors_origins}})

# Use environment variables
app.config['ENV'] = os.getenv('FLASK_ENV', 'production')
PDF_DIRECTORY = os.getenv('PDF_DIRECTORY', '/app/pdfs/')
OPEN_API_KEY = os.getenv('OPENAI_API_KEY')

home_dir = os.path.expanduser("~")
data_dir = PDF_DIRECTORY #"../pdf-data/"

# Global variables to store vectordb and chain
vectordb = None
chain = None
chat_history = []
files_dictionary = []

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.headers.add('Access-Control-Allow-Private-Network', 'true')
    print("Response headers: ", response.headers)
    return response

# Function to generate a hash for a document
#TODO: Fix error in this function
def generate_document_hash(document):
    document_content = document['content']  # Assuming document is a dictionary with a 'content' key
    return hashlib.md5(document_content.encode('utf-8')).hexdigest()

# Function to check if a document's hash exists in the vector database
def document_exists_in_vectordb(vectordb, document_hash):
    # Assuming vectordb has a method to search by metadata
    results = vectordb.search_by_metadata({'hash': document_hash})
    return len(results) > 0

#Endpoint and function to upload a pdf into data_dir
@app.route('/upload_file', methods=['POST'])
def upload_pdf():
    #check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    pdf_file = request.files['file']

    print("Uploading file: ", pdf_file.filename)
    #if user does not select file, browser also
    #submit an empty part without filename
    if pdf_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    #check if the file exists in the data directory
    if os.path.exists(data_dir + pdf_file.filename):
        print("File exists in the data directory")
        #rename the existing file with a timestamp and move it to the archive directory
        #Use the format filename-<MM-DD-YYYY>-<HH-MM-SS>.pdf for the archived file
        timestamp = time.strftime("%m-%d-%Y-%H-%M-%S")
        os.system(f"mv {data_dir + pdf_file.filename} {data_dir + 'archive/' + pdf_file.filename}-{timestamp}.pdf")

    #save the pdf file to the data directory
    pdf_file.save(data_dir + pdf_file.filename)

    #load the pdf file into the vectordb
    documents = []
    pdf_path = data_dir + pdf_file.filename
    loader = PyPDFLoader(pdf_path)
    documents.extend(loader.load())
    vectordb = split_and_embed_documents(documents)
    # vectordb.persist()
    initialize_chain()

    return jsonify({"message": "PDF uploaded and processed successfully."})

def split_and_embed_documents(documents):
    
     # Split the documents into smaller chunks
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    #chunk size should be large to get proper context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2500,
        chunk_overlap = 250,
        length_function = len,
        is_separator_regex = False
    )

    documents = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # Convert the document chunks to embedding and save them to the vector store
    vectordb = Chroma.from_documents(documents, embedding=embeddings)#, persist_directory="./data")
    # Process and add documents to the vector database if they don't already exist
    # for document in documents:
    #     document_hash = generate_document_hash(document)
    #     if not document_exists_in_vectordb(vectordb, document_hash):
    #         vectordb.add_documents([document], metadata={'hash': document_hash})
    # vectordb.persist()
    return vectordb

def load_data(data_dir):
    print("Loading data from: ", data_dir)
    #delete data directory used by ChromaDB, if it exists
    # if os.path.exists("./data"):
    #     os.system("rm -rf ./data")
    #     #recreate the data directory
    #     os.system("mkdir ./data")
    #     #change the permissions of the data directory to 777
    #     os.system("chmod 777 ./data")
    #     #change the owner of the data directory to the current user
    #     os.system(f"chown -R $(whoami) ./data")

    documents = []
    # Process a list of documents from data_dir folder
    # loop through the files_dictionary list and load the documents
    for company_name, file, status in files_dictionary:
        if file.endswith(".pdf"):
            print("Processing_file: ",file)
            #check if the file exists in the data directory
            if not os.path.exists(data_dir + file):
                print(f"File {file} does not exist in the data directory ... skipping")
                continue
            pdf_path = data_dir + file
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        # elif file.endswith('.docx') or file.endswith('.doc'):
        #     doc_path = data_dir + file
        #     loader = Docx2txtLoader(doc_path)
        #     documents.extend(loader.load())
        # elif file.endswith('.txt'):
        #     text_path = data_dir + file
        #     loader = TextLoader(text_path)
        #     documents.extend(loader.load())
            #replace the existing entry in files_dictionary with a new entry containing the company name and the status (whether file was processed or not)
            status = "Processed"
            for i, entry in enumerate(files_dictionary):
                if entry[0] == company_name and entry[1] == file:
                    entry[2]= status
                    break

    vectordb = split_and_embed_documents(documents)
    return vectordb

def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

# Initialize the Q&A chain
def initialize_chain():
    global chain
    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo'),
        retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
        return_source_documents=True,
        verbose=False
    )

def set_files_dictionary():
    global files_dictionary
    files_dictionary.append(["3P Learning", "3p-learning-2015-db.pdf","Not processed"])
    # files_dictionary.append(["ABB", "abb-2015-nomura-global-markets-research.pdf","Not processed"])
    # files_dictionary.append(["Apple Inc", "apple-inc-2010-goldman-sachs.pdf","Not processed"])
    # files_dictionary.append(["CBS Corporation", "cbs-corporation-2015-db.pdf","Not processed"])
    # files_dictionary.append(["Duke Energy", "duke-energy-2015-gs-credit-research.pdf","Not processed"])
    # files_dictionary.append(["Imperial Oil Limited", "imperial-oil-limited-2013-rbc-capital-markets.pdf","Not processed"])
    # files_dictionary.append(["Premier Foods", "premier-foods-2015-bc-credit-research.pdf","Not processed"])
    # files_dictionary.append(["Sanofi", "sanofi-2014-gs-credit-research.pdf","Not processed"])
    # files_dictionary.append(["Schneider Electric", "schneider-electric-2015-no.pdf","Not processed"])
    # files_dictionary.append(["The Walt Disney Company", "the-walt-disney-company-2015-db.pdf","Not processed"])
    # files_dictionary.append(["Virgin Money Holdings", "virgin-money-holdings-2015-gs.pdf","Not processed"])
    return files_dictionary

def internal_initialize():
    global vectordb
    with app.app_context():
        set_files_dictionary()
        vectordb = load_data(data_dir)
        initialize_chain()

    print("Data loaded and chain initialized successfully.")
    return

# Call internal_initialize when the application starts
internal_initialize()

#Endpoint to upload a pdf file
@app.route('/upload', methods=['POST'])
def upload_file():
    #check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    pdf_file = request.files['file']
    #if user does not select file, browser also
    #submit an empty part without filename
    if pdf_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    #check if the file exists in the data directory
    if os.path.exists(data_dir + pdf_file.filename):
        #rename the existing file with a name <filename>-<YYYY-MM-DD_HH-MM-SS>.pdf and move it to the archive directory
        timestamp_str = time.strftime("%Y-%m-%d_%H-%M-%S")
        os.system(f"mv {data_dir + pdf_file.filename} {data_dir + 'archive/' + pdf_file.filename + '-' + timestamp_str}") 
        
    #save the pdf file to the data directory
    pdf_file.save(data_dir + pdf_file.filename)

    #load the pdf file into the vectordb
    documents = []
    pdf_path = data_dir + pdf_file.filename
    loader = PyPDFLoader(pdf_path)
    documents.extend(loader.load())
    vectordb = split_and_embed_documents(documents)
    # vectordb.persist()
    initialize_chain()

    return jsonify({"message": "PDF uploaded and processed successfully."})

#Endpoint to get the list of companies
@app.route('/companies', methods=['GET'])
def get_companies():
    companies = []
    for company,file,status in files_dictionary:
        if status == "Processed":
            companies.append(company)
    print("Companies: ", companies)
    return jsonify(companies)

# Endpoint to load data
@app.route('/load_data', methods=['POST'])
def load_data_endpoint():
    data_dir = request.json.get('data_dir', PDF_DIRECTORY)
    load_data(data_dir)
    initialize_chain()
    return jsonify({"message": "Data loaded and chain initialized successfully."})

def get_source_documents(source_docs):
    final_source_docs = [{"document_name": doc.metadata.get("source", "Unknown"), "page_number": doc.metadata.get("page", "Unknown")} for doc in source_docs]

    #remove duplicates from final_source_docs
    final_source_docs = [dict(t) for t in {tuple(d.items()) for d in final_source_docs}]
    return final_source_docs

def generate_metrics(queries, candidates, contexts):
    url = "http://<other_app_host>:<other_app_port>/generate_metrics"
    payload = {
        "queries": queries,
        "candidates": candidates,
        "contexts": contexts
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to generate metrics: {response.text}")

def answer_with_reranking(query, chat_history, vectordb):
    reranking_url = "http://<other_app_host>:<other_app_port>/reranking_answer"
    reranking_payload = {
        "query": query,
        "chat_history": chat_history
    }
    reranking_response = requests.post(reranking_url, json=reranking_payload)
    if reranking_response.status_code == 200:
        ranked_result = reranking_response.json()
    else:
        raise Exception(f"Failed to get reranked answer: {reranking_response.text}")


# Endpoint to handle chat queries
@app.route('/generate', methods=['GET'])
def chat_endpoint():
    global chat_history

    query = request.args.get('userMessage')
    if not query:
        return jsonify({"error": "userMessage is required"}), 400

    #result = chain.invoke({"question": query, "chat_history": chat_history})
    print("Query: ", query)
    metrics = False
    rerank = False

    try:
        # Count tokens for the query and chat history
        query_tokens = count_tokens(query)
        history_tokens = sum(count_tokens(q) + count_tokens(a) for q, a in chat_history)
        total_tokens = query_tokens + history_tokens
        
        print(f"Tokens in query: {query_tokens}")
        print(f"Tokens in chat history: {history_tokens}")
        print(f"Total tokens: {total_tokens}")
    
        final_result=[]
        queries = [query]
        metrics=[]
        candidates=[]
        if rerank==True:
            print("Reranking ...")
            final_result = answer_with_reranking(query, chat_history, vectordb)
            candidates=final_result['result']
            chat_history.append((query, final_result["result"]))
            print("Ranked result: ", candidates)
        else:
            final_result = chain.invoke({"question": query, "chat_history": chat_history})        
            chat_history.append((query, final_result["answer"]))
            print("Answer: ", final_result["answer"])

        if metrics==True:
            candidates.append(candidates)
            metrics = generate_metrics(queries, candidates, get_source_documents(final_result["source_documents"]))
            return jsonify({"userMessage": query, "answer": final_result, "source_documents": get_source_documents(final_result["source_documents"]), "metrics": metrics})
        else:
            return jsonify({
                "userMessage": query,
                "answer": final_result["answer"],
                "source_documents": get_source_documents(final_result["source_documents"])
            })
        
    except Exception as e:
        print(f"Error in /generate endpoint: {e}")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(f"Exception type: {exc_type}, File name: {fname}, Line number: {exc_tb.tb_lineno}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    data_dir = PDF_DIRECTORY
    internal_initialize()
    # load_data(data_dir)
    # query = "What is the analyst rating for the company 3P Learning?"

    # #create an array of queries each of which takes company as a parameter
    # company = "3P Learning"
    # queries = [
    #     f"What is the analyst rating for the company {company}?",
    #     f'Provide a summary of the research report for {company}',
    #     f'What are the analyst ratings for {company}?',
    #     f'What are the risks for {company}?',
    #     f'What are the opportunities for {company}?',
    #     f'What is the target price for {company}?',
    #     f'How does the analyst justify the target price for {company}?',
    #     f'Who are the competitors of {company}?',
    #     f'Provide the key financial metrics used by the analyst to rate {company} and for each of these metrics provide the values',
    #     f'Provide the details of the analyst who wrote the report for {company}, their affiliation, and the date of the report.']

    # #for each query in the queries array, invoke the chain and print the result
    # unranked_candidates = []
    # unranked_contexts = []
    # ranked_candidates = []
    # ranked_contexts = []
    # for query in queries:
    #     unranked_result = chain.invoke({"question": query, "chat_history": chat_history})
    #     unranked_candidates.append(unranked_result["answer"])
    #     unranked_contexts.append([doc.page_content for doc in unranked_result['source_documents']])
    #     # result = chain.invoke({"question": query, "chat_history": chat_history})
    #     ranked_result = answer_with_reranking(query, chat_history)
    #     ranked_candidates.append(ranked_result["result"])
    #     ranked_contexts.append([doc.page_content for doc in ranked_result['source_documents']])

    # ranked_metrics = generate_metrics(queries, ranked_candidates, ranked_contexts)
    # print(ranked_metrics)
    # #save the result to a csv file
    # ranked_metrics.to_csv("ranked_metrics.csv")

    # unranked_metrics = generate_metrics(queries, unranked_candidates, unranked_contexts)
    # print(unranked_metrics)
    # #save the result to a csv file
    # unranked_metrics.to_csv("unranked_metrics.csv")

    app.run(port=5001, debug=True, host='0.0.0.0')