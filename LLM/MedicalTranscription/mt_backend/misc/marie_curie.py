from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain_community.chat_models import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_experimental.llms.ollama_functions import OllamaFunctions
import logging

def ingestion(text):

    # Convert the text into documents
    documents = [Document(page_content=text)]

    # Initialize the language model for text-to-graph conversion
    llm = ChatOllama(model="llama3", temperature=0)
    llm_transformer_filtered = LLMGraphTransformer(llm=llm)
    
    # Convert the text into graph documents
    graph_documents = llm_transformer_filtered.convert_to_graph_documents(documents)

    # Add the generated graph into Neo4j
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )
    
    # Optional: Create embeddings for more complex search queries
    embed = OllamaEmbeddings(model="mxbai-embed-large")
    vector_index = Neo4jVector.from_existing_graph(
        url="bolt://localhost:7687",
        embedding=embed,
        username="neo4j",
        password="MyNeo4J@2024",
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )
    vector_retriever = vector_index.as_retriever()

    # Return the graph documents and vector retriever for further use
    return graph_documents, vector_retriever

def extract_entities(question: str) -> list[str]:
    """Helper function to extract entities from a question using the entity_chain."""
    response = entity_chain.invoke({"question": question})
    entities = response['raw'].tool_calls[0]['args']['properties']['names']
    print("Retrieved Entities")
    print(entities)
    return entities

def querying_neo4j(question: str) -> str:
    entities = extract_entities(question)
    result = ""

    for entity in entities:
        query_response = graph.query(
            """MATCH (p:Person {id: $entity})-[r]->(e)
            RETURN p.id AS source_id, type(r) AS relationship, e.id AS target_id
            LIMIT 50""",
            {"entity": entity}
        )
        result += "\n".join([f"{el['source_id']} - {el['relationship']} -> {el['target_id']}" for el in query_response])
    
    return result

def graph_retriever(question: str) -> str:
    entities = extract_entities(question)
    result = ""

    for entity in entities:
        query_response = graph.query(
            """MATCH (p:Person {id: $entity})-[r]->(e)
            RETURN p.id AS source_id, type(r) AS relationship, e.id AS target_id
            LIMIT 50""",
            {"entity": entity}
        )
        result += "\n".join([f"{el['source_id']} - {el['relationship']} -> {el['target_id']}" for el in query_response])
    
    return result

# Define a function that combines data retrieved from both Neo4j and vector embeddings
def full_retriever(question: str):
    # Retrieve graph data for the question using the graph_retriever function
    graph_data = graph_retriever(question)
    print("Graph Data")
    print(graph_data)
    # Retrieve vector data by invoking the vector retriever with the question
    vector_data = [el.page_content for el in vector_retriever.invoke(question)]
    
    # Combine the graph data and vector data into a formatted string
    return f"Graph data: {graph_data}\nVector data: {'#Document '.join(vector_data)}"

def querying_ollama(question):
    llm = ChatOllama(model="llama3", temperature=0)

    # Define a prompt template for generating a response based on context
    template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    Answer:"""
    
    # Create a prompt from the template, which takes the context and question as input
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create a processing chain that:
    # 1. Generates context using the full_retriever function
    # 2. Passes through the question as-is using RunnablePassthrough
    # 3. Applies the prompt template to generate the final question
    # 4. Uses the LLM (language model) to generate the answer
    # 5. Uses StrOutputParser to format the output as a string
    chain = (
        {
            "context": lambda input: full_retriever(input),  # Generate context from the question
            "question": RunnablePassthrough(),  # Pass the question through without modification
        }
        | prompt  # Apply the prompt template
        | llm  # Use the language model to answer the question based on context
        | StrOutputParser()  # Parse the model's response as a string
    )

    # Test the chain with a question
    response = chain.invoke(input="Who are Marie Curie and Pierre Curie?")
    print("Final Answer")
    print(response)

if __name__ == "__main__":

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Disable Neo4j warnings
    logging.getLogger("neo4j").setLevel(logging.INFO)

    graph = Neo4jGraph(
        url= "bolt://localhost:7687" ,
        username="neo4j", #default
        password="MyNeo4J@2024" #change accordingly
    )

    text = """
    Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
    She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
    Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
    She was, in 1906, the first woman to become a professor at the University of Paris. 
    """
        
    # Define a model for the extracted entities from the text
    class Entities(BaseModel):
        names: list[str] = Field(..., description="All entities from the text")

    # Define a prompt to extract entities from the input query
    prompt = ChatPromptTemplate.from_messages([ 
        ("system", "Extract organization and person entities from the text."),
        ("human", "Extract entities from: {question}")
    ])

    # Initialize the Ollama model for entity extraction with LLM (using "llama3")
    llm = OllamaFunctions(model="llama3", format="json", temperature=0)

    # Combine the prompt and LLM to create an entity extraction chain
    # The output is structured to match the "Entities" model
    entity_chain = prompt | llm.with_structured_output(Entities, include_raw=True)

    # Ingest the text into Neo4j
    graph_documents, vector_retriever = ingestion(text)
    # Query Neo4j for relationships of the extracted entities
    result = querying_neo4j("Who are Marie Curie and Pierre Curie?")
    # Query Ollama for a response based on the extracted entities and embeddings
    querying_ollama("Who are Marie Curie and Pierre Curie?")
