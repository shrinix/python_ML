import os
import pickle

import streamlit as st
from PyPDF2 import PdfReader
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.llms.openai import OpenAI

os.environ['OPENAI_API_KEY'] = ''

def main():
    st.header('Chat with  PDF 💬')
    st.sidebar.title('LLM ChatApp using LangChain')
    st.sidebar.markdown('''
    This is an LLM powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM Model 
    ''')

    st.sidebar.write('Do checkout my YouTube Channel as well for amazing content [Muhammad Moin](https://www.youtube.com/channel/UC--6PuiEdiQY8nasgNluSOA)')

    # Upload a PDF File
    pdf = st.file_uploader("Upload your PDF File", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        #st.write(text)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len

        )
        chunks = text_splitter.split_text(text=text)
        #st.write(chunks[0])
        store_name = pdf.name[:-4]
        st.write(store_name)
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            st.write('Embeddings Loaded from the Disk')
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
            st.write('Embeddings Created')
        query = st.text_input("Ask Question from your PDF File")
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = OpenAI()
            chain = load_qa_chain(llm = llm, chain_type='stuff')
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)




if __name__ == '__main__':
    main()
