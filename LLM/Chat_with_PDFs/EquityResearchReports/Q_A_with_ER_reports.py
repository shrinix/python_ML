from langchain import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain import VectorDBQA, LLMChain
from langchain_community.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain.globals import set_verbose
from langchain.globals import set_debug
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
import streamlit as st
import sys
import os
import os.path
from langchain_community.chat_models import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain

import sys
sys.path.insert(0, '/Users/shriniwasiyengar/git/python_ML/LLM/Chat_with_PDFs/EquityResearchReports/utils')
from pdf_loaders import PdfToTextLoader #TODO: Move to utils folder
from utilities import process_files
from dataset_vectorizers import DatasetVectorizer

PDF_FILES= []
COMPANY_NAMES = []
TEXT_FILES = []
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 500

# Specify the directory for data files
home_dir = os.path.expanduser("~")
data_dir = "pdf-data/"
directory = data_dir

#set_debug(True)
class FilteredRetriever:
    def __init__(self, retriever, filter_prefix):
        self.retriever = retriever
        self.filter_prefix = filter_prefix

    def retrieve(self, *args, **kwargs):
        results = self.retriever.retrieve(*args, **kwargs)
        return [doc for doc in results if doc['source'].startswith(self.filter_prefix)]

COMPANY_NAMES, PDF_FILES, DOCX_FILES, TEXT_FILES = process_files(directory)

# Load the PDF files
for pdf_file in PDF_FILES:
        txt_file = pdf_file.replace(".pdf", ".txt")
        
        #delete old txt file if it exists
        if os.path.exists(txt_file):
            os.remove(txt_file)

        #PDFToTextLoader class is used to load the PDF file and save it as a text file
        pdf_loader = PdfToTextLoader(pdf_file, txt_file)
        text = pdf_loader.load_pdf()
        print(f"PDF file converted to TEXT successfully: {txt_file}")

dataset_vectorizer = DatasetVectorizer()
txt_documents = []
text_chunks = []
embeddings = []

for txt_path in TEXT_FILES:
    txt_document, text_chunk, embedding = dataset_vectorizer.vectorize(txt_path, chunk_size=CHUNK_SIZE,
                                                                     chunk_overlap=CHUNK_OVERLAP)
    txt_documents.append(txt_document)
    text_chunks.append(text_chunk)
    embeddings.append(embedding)

llm = ChatOllama(model='llama2', config={'max_new_tokens': 512, 'temperature': 0.00, 'context_length': 6000})

#qa_chain_1 = VectorDBQA.from_chain_type(llm=llm, chain_type='stuff', vectorstore=docsearch_1)
source_filter = TEXT_FILES[0]
memory1 = st.session_state.memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, output_key="answer"
)

custom_prompt = (
    """You are an expert equity analyst who is responsible for monitoring and maintaining a portfolio of stocks and securities. 
    Equity Research Reports on various companies are available for analysis. You need to analyze and summarize 
    the information extracted from the set of reports provided. The summary must contain the following points: 
    1. Name of the company for which the report is prepared i.e. company being analyzed. 
    2. Summary of the business areas in which the company operates. 
    3. A list of competitors for the company.
    4. A list of the company's strengths and weaknesses.
    5. The name of analyst and their recommendation on whether to buy, hold or sell the stocks of the company being analyzed.
    Provide your answers in concise, summary form.
    If there is any history of previous conversations, use it to answer.
    If you don't know the answer just answer that you don't know.
    ------\
    <ctx>
    {context}
    </ctx>
    ------
    <hs>
    {chat_history}
    </hs>
    ------
    """)

prompt = PromptTemplate(
                template=custom_prompt,
                input_variables=["context", "chat_history"])

chain = load_summarize_chain(llm, chain_type="stuff")
              search = vectordb.similarity_search(" ")
              summary = chain.run(input_documents=search, question="Write a summary within 200 words.")

# qa_chain_1 = ConversationalRetrievalChain.from_llm(llm=llm,
#                                    memory=memory1,
#                                    retriever=embeddings[0].as_retriever(search_kwargs={"k": 6,
#                                            "filter":{'source': source_filter}}),
#                                    return_source_documents=True,
#                                    verbose=False,
#                                    output_key='answer',
#                                    combine_docs_chain_kwargs={'prompt': qa_prompt})
# Instantiate chain
chain = create_stuff_documents_chain(llm, prompt)

source_filter = TEXT_FILES[1]
st.write("Files vectorized successfully.")

# ----- Write questions separated by a new line -----
chat_history_1 = []
summary_of_answers = ""
#trim the result to remove trailing spaces and newlines
result_1 = qa_chain_1.invoke({"chat_history": chat_history_1})
result_1['answer'] = result_1['answer'].strip()
summary_of_answers += f"{COMPANY_NAMES[0]} answer: " + result_1['answer']+ f";\n"

print(summary_of_answers)