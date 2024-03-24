from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
import chromadb
import hashlib
import uuid
class DatasetVectorizer_CS:

    """
        A class for vectorizing datasets.
    """
    def __init__(self):
        self.client = chromadb.Client()
        pass

    def generate_sha256_hash_from_text(self, text):
        # Create a SHA256 hash object
        sha256_hash = hashlib.sha256()
        # Update the hash object with the text encoded to bytes
        sha256_hash.update(text.encode('utf-8'))
        # Return the hexadecimal representation of the hash
        return sha256_hash.hexdigest()

    #write a function to returng a handle to client
    def get_client(self):
        return self.client

    def vectorize(self, full_path_and_txt_file_name, chunk_size=1000, chunk_overlap=500):
        documents = []

        print('Processing: '+full_path_and_txt_file_name)

        doc_loader = TextLoader(full_path_and_txt_file_name)
        documents.extend(doc_loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=chunk_overlap, chunk_size=chunk_size)
        texts = text_splitter.split_documents(documents)

        hugging_face_embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                           model_kwargs={'device': 'cpu'})

        # sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        #     model_name="all-MiniLM-L6-v2")

        # Generate embeddings for your document chunks
        document_embeddings = hugging_face_embeddings.embed_documents(full_path_and_txt_file_name)
        #for each embedding in document_embeddings, generate an id and add it to a list of ids.
        #Then use the list of ids to add the embeddings to the collection
        #Use a formula: id = text_file_path + str(i)

        id_list = [str(uuid.uuid4()) for _ in range(len(document_embeddings))]

        metadatas_list = []
        for i in range(len(document_embeddings)):
            metadatas_list.append({"source": full_path_and_txt_file_name})

        text_file_name = full_path_and_txt_file_name.split("/")[-1]
        collection = self.client.create_collection(name=text_file_name)
        collection.add(
            embeddings=document_embeddings,
            #documents=texts,
            metadatas=metadatas_list,
            ids=id_list
        )

        #docsearch = Chroma.from_documents(texts, embeddings)

        return documents, texts, collection