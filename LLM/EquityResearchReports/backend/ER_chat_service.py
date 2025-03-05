# __import__('pysqlite3')
import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import os
import argparse
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import tiktoken
import hashlib

#check if OPEN_AI_API_KEY is set
if 'OPENAI_API_KEY' not in os.environ:
    print("Please set the OPENAI_API_KEY environment variable")
    sys.exit(1)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Start the Flask application with a specific .env file.')
parser.add_argument('--env-file', type=str, required=False, help='Path to the .env file')
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

@app.after_request
def after_request(response):
    # response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.headers.add('Access-Control-Allow-Private-Network', 'true')
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
    #if user does not select file, browser also
    #submit an empty part without filename
    if pdf_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    #save the pdf file to the data directory
    pdf_file.save(data_dir + pdf_file.filename)

    #load the pdf file into the vectordb
    documents = []
    pdf_path = data_dir + pdf_file.filename
    loader = PyPDFLoader(pdf_path)
    documents.extend(loader.load())
    vectordb = split_and_embed_documents(documents)
    vectordb.persist()
    initialize_chain()

    return jsonify({"message": "PDF uploaded and processed successfully."})

def split_and_embed_documents(documents):
    
     # Split the documents into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    documents = text_splitter.split_documents(documents)

    # Convert the document chunks to embedding and save them to the vector store
    vectordb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(), persist_directory="./data")
    # Process and add documents to the vector database if they don't already exist
    # for document in documents:
    #     document_hash = generate_document_hash(document)
    #     if not document_exists_in_vectordb(vectordb, document_hash):
    #         vectordb.add_documents([document], metadata={'hash': document_hash})
    vectordb.persist()
    return vectordb

def load_data(data_dir):
    print("Loading data from: ", data_dir)
    #delete data directory if it exists
    if os.path.exists("./data"):
        os.system("rm -rf ./data")

    documents = []
    # Process a list of documents from data_dir folder
    for file in os.listdir(data_dir):
        if file.endswith(".pdf"):
            print("Processing_file: ",file)
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

def internal_initialize():
    global vectordb
    with app.app_context():
        vectordb = load_data(data_dir)
        initialize_chain()

# Call internal_initialize when the application starts
internal_initialize()

# Endpoint to load data
@app.route('/load_data', methods=['POST'])
def load_data_endpoint():
    data_dir = request.json.get('data_dir', "../pdf-data/")
    load_data(data_dir)
    initialize_chain()
    return jsonify({"message": "Data loaded and chain initialized successfully."})

# Endpoint to handle chat queries
@app.route('/generate', methods=['GET'])
def chat_endpoint():
    global chat_history

    query = request.args.get('userMessage')
    if not query:
        return jsonify({"error": "userMessage is required"}), 400

    #result = chain.invoke({"question": query, "chat_history": chat_history})
    print("Query: ", query)

    # Count tokens for the query and chat history
    query_tokens = count_tokens(query)
    history_tokens = sum(count_tokens(q) + count_tokens(a) for q, a in chat_history)
    total_tokens = query_tokens + history_tokens
    
    print(f"Tokens in query: {query_tokens}")
    print(f"Tokens in chat history: {history_tokens}")
    print(f"Total tokens: {total_tokens}")
 
    result = chain.invoke(
        {"question": query, "chat_history": chat_history})
    
    #print the source documents used to obtain the answer
    print("Source documents used in the answer: ", result["source_documents"])
    
    chat_history.append((query, result["answer"]))
    print("Answer: ", result["answer"])

    return jsonify({"userMessage": query, "answer": result["answer"]})

if __name__ == '__main__':
    # data_dir = "../pdf-data/"
    # load_data(data_dir)
    #chat("What is the analyst rating for the company Duke Energy?")
    #chat("Create a list of all companies alongwith their respective analyst rating and the name of the analyst providing the rating")
    
    # app.run(host='0.0.0.0', port=5000, debug=True) #use this only for remote server
    app.run(port=5001, debug=True, host='0.0.0.0')