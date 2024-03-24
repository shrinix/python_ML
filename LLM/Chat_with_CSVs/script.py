from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import sys
from langchain_community.chat_models import ChatOllama

DB_FAISS_PATH = "vectorstore/db_faiss"
loader = CSVLoader(file_path="data/2019.csv", encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()
print(data)

# Split the text into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)

print(len(text_chunks))

# Download Sentence Transformers Embedding From Hugging Face
embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

# COnverting the text Chunks into embeddings and saving the embeddings into FAISS Knowledge Base
docsearch = FAISS.from_documents(text_chunks, embeddings)

docsearch.save_local(DB_FAISS_PATH)

#query = "What is the value of GDP per capita of Finland provided in the data?"

#docs = docsearch.similarity_search(query, k=3)

#print("Result", docs)

# llm = CTransformers(model="../models/llama-2-7b-chat.ggmlv3.q8_0.bin",
#                     model_type="llama",
#                     config={'max_new_tokens': 512,
#                             'temperature': 0.001,
#                             'context_length': 5000})
llm = ChatOllama(model='llama2', config={'max_new_tokens': 512, 'temperature': 0.01, 'context_length': 6000})

qa = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())

while True:
    chat_history = []
    #query = "What is the value of  GDP per capita of Finland provided in the data?"
    #query = Can you list the bottom 3 countries with the lowest GDP per capita?"
    query = input(f"Input Prompt: ")
    if query == 'exit':
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    result = qa.invoke({"question":query, "chat_history":chat_history})
    print("Response: ", result['answer'])