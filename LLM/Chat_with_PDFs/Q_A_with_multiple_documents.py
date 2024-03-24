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
from pdf_loaders import PdfToTextLoader

from dataset_vectorizers import DatasetVectorizer
import streamlit as st
import sys
import os
import os.path
from langchain_community.chat_models import ChatOllama

#set_debug(True)
class FilteredRetriever:
    def __init__(self, retriever, filter_prefix):
        self.retriever = retriever
        self.filter_prefix = filter_prefix

    def retrieve(self, *args, **kwargs):
        results = self.retriever.retrieve(*args, **kwargs)
        return [doc for doc in results if doc['source'].startswith(self.filter_prefix)]

PDFS = []
NAMES = []
TXTS = []
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 500
# Specify the directory
directory = "data/"

# Get list of all files in the directory which match the pattern *.pdf
file_match = ".pdf"
if os.path.exists(directory):
    file_names = [f for f in os.listdir(directory) if f.endswith(file_match)]
    if not file_names:
        print("No PDF files found in the directory| "+directory)
else:
    print("The directory does not exist: "+directory)
    sys.exit()

dataset_vectorizer = DatasetVectorizer()
txt_documents = []
text_chunks = []
embeddings = []

for file_name in file_names:
    #print(documents)
    full_path_and_pdf_file_name = directory + file_name
    PDFS.append(full_path_and_pdf_file_name)
    #Extract name of candidate from resume filename and add to NAMES list
    #the name of the file has he format <First Name>_<Last Name>_Resume.pdf
    #the extracted candidate name should be in the format: <First Name> <Last Name>
    name = file_name.split("_")[0] + " " + file_name.split("_")[1]
    NAMES.append(name)
    txt_path = file_name.replace(".pdf", ".txt")
    full_path_and_txt_file_name = directory + txt_path
    #delete old txt file if it exists
    if os.path.exists(full_path_and_txt_file_name):
        os.remove(full_path_and_txt_file_name)
    pdf_loader = PdfToTextLoader(full_path_and_pdf_file_name, full_path_and_txt_file_name)
    text = pdf_loader.load_pdf()
    TXTS.append(full_path_and_txt_file_name)
    st.write("Files loaded successfully.")

for txt_path in TXTS:
    txt_document, text_chunk, embedding = dataset_vectorizer.vectorize(txt_path, chunk_size=CHUNK_SIZE,
                                                                     chunk_overlap=CHUNK_OVERLAP)
    txt_documents.append(txt_document)
    text_chunks.append(text_chunk)
    embeddings.append(embedding)

#To search for similar documents, we can use the similarity_search method of the vector store. This method takes a query and returns the most similar documents in the store. The query can be a string or a list of strings.
#The number of similar documents to return can be specified using the k parameter. The default value of k is 5.
#docs = vector_store.similarity_search(query)

#vector_store=FAISS.from_documents(text_chunks, embeddings)
# llm=CTransformers(model="models//llama-2-7b-chat.ggmlv3.q8_0.bin",
#                   model_type="llama",
#                   config={'max_new_tokens':512,
#                           'temperature':0.01,
#                           'context_length': 5000})

llm = ChatOllama(model='llama2', config={'max_new_tokens': 512, 'temperature': 0.01, 'context_length': 6000})

#qa_chain_1 = VectorDBQA.from_chain_type(llm=llm, chain_type='stuff', vectorstore=docsearch_1)
source_filter = TXTS[0]
memory1 = st.session_state.memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, output_key="answer"
)
memory2 = st.session_state.memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, output_key="answer"
)

custom_prompt = (
    "You are a human resource specialist who is responsible for "
    "reviewing candidates' CVs. You will be given the CV of the "
    "candidate and your job is to extract the information based "
    "on questions mentioned below."
    "Use the following context (delimited by <ctx></ctx>) to answer the questions."
    "If there is any history of previous conversations, use it to answer "
    "(delimited by <hs></hs>)"
    "If you don't know the answer just answer that you don't know." 
    "------"
    "<ctx>"
    "{context}"
    "</ctx>"
    "------"
    "<hs>"
    "{chat_history}"
    "</hs>"
    "------"
    "Question:"
    "{question}")

qa_prompt = PromptTemplate(
                template=custom_prompt,
                input_variables=["context", "chat_history", "question"])
qa_chain_1 = ConversationalRetrievalChain.from_llm(llm=llm,
                                   memory=memory1,
                                   retriever=embeddings[0].as_retriever(search_kwargs={"k": 6,
                                           "filter":{'source': source_filter}}),
                                   return_source_documents=True,
                                   verbose=False,
                                   output_key='answer',
                                   combine_docs_chain_kwargs={'prompt': qa_prompt})

#qa_chain_2 = VectorDBQA.from_chain_type(llm=llm, chain_type='stuff', vectorstore=docsearch_2)
source_filter = TXTS[1]
#filtered_retriever = FilteredRetriever(embeddings[0].as_retriever(),filter_prefix=source_filter)
qa_chain_2 = ConversationalRetrievalChain.from_llm(llm=llm,
                                   memory=memory2,
                                   retriever=embeddings[1].as_retriever(search_kwargs={"k": 6,
                                           "filter":{'source': source_filter}}),
                                   return_source_documents=True,
                                   verbose=False,
                                   combine_docs_chain_kwargs={'prompt': qa_prompt})
st.write("Files vectorized successfully.")

  # ----- Write questions separated by a new line -----
st.header("Write the questions to generate a summary")

st.write("The questions should be separated by a new line.")
questions = st.text_area("Questions", value="""
What are the qualifications of the candidate?
What is the work experience of the candidate?
What universities has the candidate has studied at?
What companies the candidate has worked for?
What skills does the candidate have?""")
QUESTIONS = questions.split("\n")
QUESTIONS = [q.strip() for q in QUESTIONS if len(q) > 0]

# ----- Select final criteria for decision-making -----
st.header("Select the final criteria for decision-making")
st.write("The criteria should be separated by a new line.")
criteria = st.text_area("Criteria", value="""
1. Skills of candidate
2. Length of work experience
3. Experience in working at financial companies""")
CRITERIA = criteria.split("\n")
CRITERIA = [c.strip() for c in CRITERIA if len(c) > 0]
final_criteria = "".join([f"{i}. {c}\n" for i, c in enumerate(CRITERIA, 1)])

# ----- Generate the intermediate answers for the document summary -----
summary_of_answers = ""
for q in QUESTIONS:
    print(q)
    chat_history_1 = []
    chat_history_2 = []
    #trim the result to remove trailing spaces and newlines
    result_1 = qa_chain_1.invoke({"question": q, "chat_history": chat_history_1})
    result_1['answer'] = result_1['answer'].strip()
    result_2 = qa_chain_2.invoke({"question": q, "chat_history": chat_history_2})
    result_2['answer'] = result_2['answer'].strip()
    summary_of_answers += "Question: " + q + "\n"
    #result_1['source_documents'] returns a list of references[0]['text']
    summary_of_answers += f"{NAMES[0]} answer: " + result_1['answer']+ f";\n {NAMES[1]} answer: " + result_2['answer'] + "\n"
print(summary_of_answers)
#
# template=("You are an expert recruiter. You will use the below information"
#  "extracted from the CVs of two candidates to help hire a candidate for a "
#  "data science project at a large financial company."
#  "{summary_of_answers}"
#  "I want you to tell me which candidate is better and why."
#  "Give me a rating (x out of 10) for the following categories for each candidate "
#  "separately with a short explanation (10 words max) for each category."
#  "{final_criteria}"
#  "You need to provide a final recommendation for the candidate to be selected, based on the"
#  "above ratings.")
# prompt = PromptTemplate(
#         input_variables=["summary_of_answers", "final_criteria"],
#         template=template,)
#
# #start=timeit.default_timer()
# chain = LLMChain(llm=llm, prompt=prompt)
# answer = chain.run({"summary_of_answers": summary_of_answers, "final_criteria": final_criteria})
#
# # ----- Generate the final answer -----
# st.header("Final answer")
# st.write(answer)
# print(answer)
#end=timeit.default_timer()
