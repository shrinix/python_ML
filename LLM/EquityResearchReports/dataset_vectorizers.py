from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader

class DatasetVectorizer:
    """
        A class for vectorizing datasets.
    """
    def __init__(self):
        pass

    def vectorize(self, text_file_path, chunk_size=1000, chunk_overlap=500):
        documents = []
        print('Processing: '+text_file_path)

        doc_loader = TextLoader(text_file_path)
        documents.extend(doc_loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=chunk_overlap, chunk_size=chunk_size)
        texts = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                           model_kwargs={'device': 'cpu'})
        docsearch = Chroma.from_documents(texts, embeddings)

        return documents, texts, docsearch