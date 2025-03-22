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
import json
from fpdf import FPDF

#set environment variable to disable ChromadB telemetry
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
global logger

cache = {}
def generate_cache_key(query: str) -> str:
        """
        Generate a unique cache key for a given query.
        
        Args:
            query (str): The query string.
        
        Returns:
            str: The cache key.
        """
        return hashlib.md5(query.encode()).hexdigest()

def logging_setup():
    LOGS_DIRECTORY = os.getenv('LOGS_DIRECTORY', '/app/logs/')

    # Set up logging
    log_dir = os.path.join(os.path.dirname(__file__), LOGS_DIRECTORY)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    # Ensure the logs directory exists

    global logger
    logger = logging.getLogger(__name__)
    log_filename = os.path.join(log_dir, f"ER-chat-service-{time.strftime('%Y-%m-%d_%H-%M-%S')}.log")
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Redirect stdout and stderr to the logger
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    sys.stdout = open(log_filename, 'a', buffering=1)
    sys.stderr = sys.stdout

    logger.info(f"Logging setup completed: {log_dir}")

logging_setup()

# Set display options to control wrapping
pd.set_option('display.max_columns', None)  # Do not truncate the list of columns
pd.set_option('display.max_rows', None)     # Do not truncate the list of rows
pd.set_option('display.width', 1000)        # Set the display width to a large value
pd.set_option('display.max_colwidth', 50)   # Set the maximum column width
pd.set_option('display.colheader_justify', 'left')  # Justify column headers to the left
logger.info(f"Virtual environment: {os.getenv('VIRTUAL_ENV')}")
logger.info(f"Python path: {os.getenv('PYTHONPATH')}")
#logger.info versions of langchain and langchain_openai
# logger.info("Langchain OpenAI version: ", langchain_openai.__version__)
    

#check if OPEN_AI_API_KEY is set
if 'OPENAI_API_KEY' not in os.environ:
    logger.info("Please set the OPENAI_API_KEY environment variable.")
    sys.exit(1)

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Parse command line arguments
parser = argparse.ArgumentParser(description='Start the Flask application with a specific .env file.')
parser.add_argument('--env_file', type=str, required=False, help='Path to the .env file')
try:
    args = parser.parse_args()
except SystemExit as e:
    if e.code == 2:  # Error code 2 indicates invalid arguments
        logger.info("No valid arguments provided. Using default settings.")
        args = argparse.Namespace(env_file=None)

if args.env_file:
    logger.info(f"Loading environment variables from {args.env_file}")
    # Load environment variables from the specified .env file
    load_dotenv(args.env_file)
else:
    logger.info("No .env file specified. Loading environment variables from backend.env")
    # Load environment variables from .env file
    load_dotenv('backend.env')

app = Flask(__name__)
# Get CORS origins from environment variables
cors_origins = os.getenv('CORS_ORIGINS', '*')

# # Enable CORS for specific IP address
CORS(app,supports_credentials=True, resources={r"/*": {"origins": cors_origins}})

# Use environment variables
app.config['ENV'] = os.getenv('FLASK_ENV', 'development')
PDF_DIRECTORY = os.getenv('PDF_DIRECTORY', '/app/pdfs/')
output_dir = os.getenv('OUTPUT_DIR', '/app/output/')
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
    logger.info(f"Response headers: {response.headers}")
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

    logger.info(f"Uploading file: {pdf_file.filename}")
    #if user does not select file, browser also
    #submit an empty part without filename
    if pdf_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    #check if the file exists in the data directory
    if os.path.exists(data_dir + pdf_file.filename):
        logger.info("File exists in the data directory")
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
    logger.info(f"Loading data from: {data_dir}")
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
            logger.info(f"Processing_file: {file}")
            #check if the file exists in the data directory
            if not os.path.exists(data_dir + file):
                logger.info(f"File {file} does not exist in the data directory ... skipping")
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

    logger.info("Data loaded and chain initialized successfully.")
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
    logger.info(f"Companies: {companies}")
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
    # Get the URL from the environment variable, with a fallback to localhost for local development
    url = os.getenv("METRICS_SERVICE_URL", "http://localhost:5002/generate_metrics")
    payload = {
        "queries": queries,
        "candidates": candidates,
        "contexts": contexts
    }
    logger.info("Inovking metrics endpoint ...")
    logger.info(f"Payload: {payload}")
    metrics_df = None
    try:
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
        raise
    except Exception as err:
        logger.error(f"Other error occurred: {err}")
        raise
    if response.status_code == 200:
        logger.info("Metrics generated successfully.")
        logger.info(f"Metrics response: {response.text}")
        #convert json response to a pandas dataframe
         # Parse the JSON response
        response_json = response.json()
        # Convert the metrics data back to a pandas DataFrame
        metrics_df = pd.json_normalize(response_json)
        logger.info("Metrics DataFrame:")
        if metrics_df is not None:
            logger.info(metrics_df)
        else:
            logger.info("Metrics DataFrame is None.")
    else:
        raise Exception(f"Failed to generate metrics: {response.text}")
    
    return metrics_df

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
    logger.info(f"Query: {query}")
    calc_metrics = True
    rerank = False

    try:
        # Count tokens for the query and chat history
        query_tokens = count_tokens(query)
        history_tokens = sum(count_tokens(q) + count_tokens(a) for q, a in chat_history)
        total_tokens = query_tokens + history_tokens
        
        logger.info(f"Tokens in query: {query_tokens}")
        logger.info(f"Tokens in chat history: {history_tokens}")
        logger.info(f"Total tokens: {total_tokens}")

        final_result=[]
        queries = query
        metrics=[]
        candidates=[]
        contexts=[]
    
        cache_key = generate_cache_key(query)
        if cache_key in cache:
            logger.info("Using cached response ...")
            final_result = cache[cache_key]
        else:
            logger.info("Cache miss ...")
            if rerank==True:
                logger.info("Reranking ...")
                final_result = answer_with_reranking(query, chat_history, vectordb)
                candidates=final_result['result']
                chat_history.append((query, final_result["result"]))
                logger.info(f"Ranked result: {candidates}")
            else:
                final_result = chain.invoke({"question": query, "chat_history": chat_history})        
                chat_history.append((query, final_result["answer"]))
                logger.info(f"Answer: {final_result['answer']}")

            # Cache the response
            cache[cache_key] = final_result

        if calc_metrics==True:
        
            logger.info("Invoking metrics endpoint ...")
            if rerank==True:
                candidates=final_result['result']
            else:
                candidates=final_result['answer']
            
            contexts.append([doc.page_content for doc in final_result['source_documents']])
            queries, candidates, contexts = [queries], [candidates], contexts
            metrics = generate_metrics(queries, candidates, contexts)
            logger.info(f"Metrics response: {metrics}")
            return jsonify({
                "userMessage": query,
                "answer": final_result["answer"],
                "source_documents": get_source_documents(final_result["source_documents"]),
                "metrics": metrics.to_dict(orient='records')  # Convert DataFrame to a list of dictionaries
            })
        else:
            return jsonify({
                "userMessage": query,
                "answer": final_result["answer"],
                "source_documents": get_source_documents(final_result["source_documents"])
             })
        
    except Exception as e:
        logger.info(f"Error in /generate endpoint: {e}")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.info(f"Exception type: {exc_type}, File name: {fname}, Line number: {exc_tb.tb_lineno}")
        return jsonify({"error": str(e)}), 500

# Endpoint to handle chat queries
@app.route('/generate_IA_report', methods=['GET'])
def generate_investment_analysis_report(company):
    system_prompt = "You are an AI assistant helping with investment analysis. Provide concise and accurate responses based on the research reports."

    #create a structure for an investment analysis template with a placeholder
    business_model = []
    market_position = []
    growth_strategy = []
    pricing_power = []
    financial_performance = []
    industry_trends = []
    competitive_landscape = []
    transaction_comparables = []
    investment_report = {
        "1. **Business Model**": business_model,
        "2. **Market Position**": market_position,
        "3. **Growth Strategy**": growth_strategy,
        "4. **Pricing Power**": pricing_power,
        "5. **Financial Performance**": financial_performance,
        "6. **Industry Trends**": industry_trends,
        "7. **Competitive Landscape**": competitive_landscape,
        "8. **Transaction Comparables**": transaction_comparables
    }

    #Some of these queries may need creation of agents to retrieve realtime information from the internet.
    queries = [
            f" Evaluate the scalability, cash generation, and capital-light nature of {company}'s business model.?",
            f'What is the market position of the {company} in local and global markets?',
            f'Evaluate the growth strategy of {company} based on factors like expanding the client base, increasing market penetration, cross-selling products, and utilizing pricing levers',
            f'Assess the pricing power of {company} based on factors like customer base fragmentation, product quality, and barriers to entry',
            f"Provide details on {company}'s financial performance and valuation.",
            f"What are the current industry trends in the {company}'s line of business and how is {company} positioned to capitalize on these trends?",
            f'How does the analyst justify the target price for {company}?',
            f'Who are the competitors of {company} and how do their metrics compare with that of {company}?',
            f'Analyze recent transactions in the space of {company} and compare them with the valuation of {company}',
    ]

    # Create a mapping of the queries to the corresponding sections in the investment report
    queries_section_mapping = {
        queries[0]:"1. **Business Model**",
        queries[1]:"2. **Market Position**",
        queries[2]:"3. **Growth Strategy**",
        queries[3]:"4. **Pricing Power**",
        queries[4]:"5. **Financial Performance**",
        queries[5]:"6. **Industry Trends**",
        queries[6]:"7. **Competitive Landscape**",
        queries[7]:"8. **Transaction Comparables**"
    }
    # Create a ConversationalRetrievalChain with a ChatOpenAI model and a VectorDB retriever
    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo'),
        retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
        return_source_documents=True,
        verbose=False)
     
    # Loop through the section_queries_mapping and populate the investment report template
    for query, section in queries_section_mapping.items():
        cache_key = generate_cache_key(query)
        if cache_key in cache:
            logger.info("Using cached response ...")
            result = cache[cache_key]
        else:
            logger.info("Cache miss ...")
            result = chain.invoke({"question": query, "chat_history": chat_history})
            chat_history.append((query, result["answer"]))
            # Cache the response
            cache[cache_key] = result

        # Wrap sentences to fit a page width
        import textwrap
        def custom_wrap(text, width):
            lines = []
            for paragraph in text.split("\n"):
                if paragraph.strip().startswith(tuple("1234567890.")):
                    lines.append(paragraph.strip())
                else:
                    lines.extend(textwrap.wrap(paragraph, width=width))
            return "\n".join(lines)

        wrapped_answer = custom_wrap(result["answer"], width=80)
        investment_report[section] = wrapped_answer

    logger.info("Investment Analysis Report Generated:")
    # logger.info(json.dumps(investment_report, indent=4))

    return jsonify({"IA_report":investment_report})

def create_investment_report_PDF(investment_report, output_dir):
    # #create a pdf file with the investment report
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Investment Analysis Report", ln=True, align='C')
    for section, content in investment_report.items():
        pdf.set_font("Arial", style="B", size=12)
        pdf.multi_cell(0, 10, txt=section, align='L')
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=content, align='L')
        pdf.ln(5)  # Add spacing between sections
    pdf.output(f"{output_dir}investment_analysis_report.pdf")
    return

if __name__ == '__main__':
    data_dir = PDF_DIRECTORY
    internal_initialize()

    # investment_report = generate_investment_analysis_report("3P Learning")
    # #save the investment report to a txt file
    # with open(f"{output_dir}investment_analysis_report.txt", "w") as f:
    #     for section, content in investment_report.items():
    #         f.write(f"{section}\
    #         \n{content}\n\n")

    # create_investment_report_PDF(investment_report, output_dir)

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

    # #for each query in the queries array, invoke the chain and logger.info the result
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
    # logger.info(ranked_metrics)
    # #save the result to a csv file
    # ranked_metrics.to_csv("ranked_metrics.csv")

    # unranked_metrics = generate_metrics(queries, unranked_candidates, unranked_contexts)
    # logger.info(unranked_metrics)
    # #save the result to a csv file
    # unranked_metrics.to_csv("unranked_metrics.csv")

    app.run(port=5001, debug=True, host='0.0.0.0')