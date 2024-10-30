from langchain import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain import VectorDBQA, LLMChain
from langchain_community.llms import CTransformers
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.globals import set_verbose
from langchain.globals import set_debug
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from typing import List
import sys
import os
import os.path
from langchain_community.chat_models import ChatOllama
from graph import GraphBuilder
from neo4j import GraphDatabase
from evaluate_RAG import RAGEvaluator
from neo4j_queries import get_node_details, get_relationship_types
import logging
import hashlib

os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "MyNeo4J@2024"

# # Add the utils directory to sys.path
# utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
# if utils_path not in sys.path:
#     sys.path.append(utils_path)

# from utilities import process_files
from graph_rag import GraphRAG

# Set the logging level to ERROR to suppress warnings
logging.getLogger("neo4j").setLevel(logging.ERROR)

PDF_FILES= []
CANDIDATE_NAMES = []
TEXT_FILES = []
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 500
# Specify the directory
home_dir = os.path.expanduser("~")
resume_dir = "git/python_ML/LLM/Chat_with_PDFs/Resume_Processor/Resumes/"
directory = home_dir + "//"+resume_dir
cache = {}

#set_debug(True)
def load_resume_files_into_graph():
    # # Load the PDF files
    # for pdf_file in PDF_FILES:
    #     txt_file = pdf_file.replace(".pdf", ".txt")
        
    #     #delete old txt file if it exists
    #     if os.path.exists(txt_file):
    #         os.remove(txt_file)

    #     #PDFToTextLoader class is used to load the PDF file and save it as a text file
    #     pdf_loader = PdfToTextLoader(pdf_file, txt_file)
    #     text = pdf_loader.load_pdf()
    #     print(f"PDF file converted to TEXT successfully: {txt_file}")

    print("Building graph from content")
    graph_builder = GraphBuilder()
    graph_builder.graph_text_documents(TEXT_FILES)
    graph_builder.index_graph()
    print("Graph built successfully.")

def generate_cache_key(query: str) -> str:
        """
        Generate a unique cache key for a given query.
        
        Args:
            query (str): The query string.
        
        Returns:
            str: The cache key.
        """
        return hashlib.md5(query.encode()).hexdigest()

def get_response(chat_history, question: str) -> str:
    """
    For the given question will formulate a search query and use a custom GraphRAG retriever 
    to fetch related content from the knowledge graph. 

    Args:
        question (str): The question posed by the user for this graph RAG

    Returns:
        str: The results of the invoked graph based question
    """
    rag = GraphRAG()
    search_query = rag.create_search_query(chat_history, question)

    template = """You are an expert medical assistant who is responsible for reviewing patient transcriptions based on the
    questions given below. 
    
    A typical patient transcription consists of the following sections: 
    1. Past Medical History
    2. Surgical History
    3. Social History
    4. Family History
    5. Current Medications
    6. Allergies
    7. Miscellaneous/Eating History
    7. Review of Systems
    8. Physical Examination

    Answer each of the following question based only on the following context:
    {context}

    Question: {question}
    Use natural language and be crisp and concise.
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    context = rag.retriever(search_query)
    chain = (
        RunnableParallel(
            {
                "context": lambda x: context,
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    cache_key = generate_cache_key(question)
    if cache_key in cache:
        return cache[cache_key], context

    result = chain.invoke({"chat_history": chat_history, "question": question})

    # Cache the response
    cache[cache_key] = result
    
    # Using invoke method to get response
    return result, context

if __name__ == "__main__":
    graph = Neo4jGraph()

    # graph_builder = GraphBuilder()
    
    #Uncomment the below line to reset the graph if a new graph is to be built
    # graph_builder.reset_graph()

    # CANDIDATE_NAMES, PDF_FILES, DOCX_FILES, TEXT_FILES = process_files(directory)

    #Uncomment the below line to reset the graph if a new graph is to be built
    # load_resume_files_into_graph()
    driver = GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]))
    nodes = get_node_details(driver)
    print("Nodes----------------")
    for node in nodes:
        # Filter out the '__Entity__' label and print the remaining labels
        filtered_labels = [label for label in node["labels"] if label != '__Entity__']
        for label in filtered_labels:
            print("Label:", label)

    relationship_types = get_relationship_types(driver)
    print("Relationships----------------")
    for relationship_type in relationship_types:
        print(relationship_type)

    driver.close()
    #wait for user input
    # input("Press Enter to continue...")

    #Ollama-based chat works only with the openhermes model. the context length needs to be set to a large value like 25000 to avoid
    #EOF errors in calling neo4j procedures.

    #UseLower temperature for more deterministic output
    #Use top-p sampling to control randomness
    # llm = ChatOllama(model='llama3', config={'max_new_tokens': 1024, 'temperature': 0.0, 'top_p':1.0, 'seed':42, 'context_length': 25000})
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, top_p=1.0, seed=42)

    questions = """
    What is the past medical history of the patient?
    What is the surgical history of the patient?"""

    QUESTIONS = questions.split("\n")
    QUESTIONS = [q.strip() for q in QUESTIONS if len(q) > 0]

    # references = ["Shriniwas Iyengar has a Bachelor of Science in Electronics Engineering from Bangalore University and a PhD in Computer Science from the University of Mumbai. He also has a Professinal Certificate in" + 
    #               "Artificial Intelligence and Machine Learning from MIT.",
    #               "Shriniwas Iyengar is a visionary and results-driven Senior Vice President with extensive experience in the financial services industry, seeking a Director of Research, Gen AI role. Proven expertise in" +
    #               "architecture, design, and development of large-scale, business-critical applications for major banks. Demonstrated success in driving growth, innovation, and operational efficiency by overseeing SDLC "+
    #               "processes and aligning technology initiatives with organizational goals. Skilled mentor and team-builder with hands-on expertise in application design, software development, and automated testing",
    #               "Shriniwas Iyengar has studied at University of Bangalore and University of Mumbai",
    #               "Shriniwas Iyengar has worked at BNY Mellon (8 years), Accenture (9 years), and Randomwalk Computing (1.5 years).",
    #               "Shriniwas Iyengar has skills in Java, Angular, Springboot Microservices, Spring Cloud, Kafka, Camunda, Python, "+
    #               "Neural Networks, Deep Learning, NLP, GenAI, Langchain, Project Management, Agile Scrum, SDLC, Software Architecture and Design, and Application Management",
    #               "Shriniwas Iyengar has studied at University of Bangalore and University of Mumbai"]

    # ----- Generate the intermediate answers for the document summary -----
    summary_of_answers = ""
    chat_history = [
            AIMessage(content="")
    ]

    question_number = 0
    answer3 = None
    answer5 = None
    # for candidate in CANDIDATE_NAMES:
    for q in QUESTIONS:
        user_query = q #q.replace("{CANDIDATE}", candidate)
        print(user_query)
        answer, context = get_response(chat_history,user_query)
        summary_of_answers += "\nQuestion: " + user_query + "\n"
        #result_1['source_documents'] returns a list of references[0]['text']
        summary_of_answers += f"answer: " + answer + "\n"
        chat_history.append(HumanMessage(content=user_query))
        chat_history.append(AIMessage(content=answer))

        # evaluator = RAGEvaluator()
        # response = answer
        # reference = references[QUESTIONS.index(q)]
        # results = evaluator.evaluate_all(user_query, response, reference)
        # summary_of_answers += str(results) + "\n"

        #assert that the answers to the 3rd and 6th questions are the same
        # if question_number == 2: #3rd question
        #     answer3 = answer
        # if question_number == 5: #6th question
        #     answer5 = answer
        # if answer3 is not None and answer5 is not None:
        #     assert answer3 == answer5, "Answers to the 3rd and 6th questions are not the same"

        question_number+=1

    print(summary_of_answers)