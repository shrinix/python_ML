import asyncio

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm.openai_llm import OpenAILLM

from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.indexes import create_fulltext_index, create_vector_index

from neo4j_graphrag.retrievers import VectorRetriever
import json
import logging

def list_indexes(driver):
    with driver.session() as session:
        result = session.run("SHOW INDEXES")
        indexes = [record for record in result]
        return indexes
    
def delete_index(driver, index_name):
    # List all indexes
    indexes = list_indexes(driver)
    #checkf if the index exists
    index_exists = False
    for index in indexes:
        if index_name in index:
            index_exists = True
            break

    if index_exists:
        # Delete the vector index
        with driver.session() as session:
            session.run(f"DROP INDEX `{index_name}`")
        print("Index deleted."+index_name)
    else:
        print("Index does not exist."+index_name)

def generate_and_store_embeddings(embedding_model, driver, text):
    # Generate embeddings for the text
    embedding = embedding_model.embed(text)

    # Use the embedding in your query or indexing process
    # Example: Store the embedding in the Neo4j database
    with driver.session() as session:
        session.run(
            """
            MERGE (c:Chunk {text: $text})
            SET c.embedding = $embedding
            """,
            text=text,
            embedding=embedding,
        )

# Run the pipeline on a piece of text
def build_graph(driver, text, entities, relations, potential_schema, embedder,llm):
    # Instantiate the SimpleKGPipeline
    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        entities=entities,
        relations=relations,
        potential_schema=potential_schema,
        on_error="IGNORE",
        from_pdf=False,
    )
    # Run the pipeline and handle JSON parsing errors
    asyncio.run(kg_builder.run_async(text=text))
    
def delete_all_graphs(driver):
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

if __name__ == "__main__":

    # Define the name of the index
    VECTOR_INDEX_NAME = "medical_vector_index"
    FULLTEXT_INDEX_NAME = "medical_fulltext_index"

    NEO4J_URI = "neo4j://localhost:7687"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "MyNeo4J@2024"

    # Instantiate the LLM
    llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})

    # text = (
    #     "The son of Duke Leto Atreides and the Lady Jessica, Paul is the heir of House "
    #     "Atreides, an aristocratic family that rules the planet Caladan."
    # )

    text = (
        "Medical transcription for Patient Number 001 has following sections:"
        "PAST MEDICAL HISTORY: He has difficulty climbing stairs, difficulty with airline seats, tying shoes.")
    long_text = ("PAST MEDICAL HISTORY:, He has difficulty climbing stairs, difficulty with airline seats, tying shoes, used to public seating, and lifting objects off the floor.  He exercises three times a week at home and does cardio.  He has difficulty walking two blocks or five flights of stairs.  Difficulty with snoring.  He has muscle and joint pains including knee pain, back pain, foot and ankle pain, and swelling.  He has gastroesophageal reflux disease.,PAST SURGICAL HISTORY:, Includes reconstructive surgery on his right hand thirteen years ago.  ,SOCIAL HISTORY:, He is currently single.  He has about ten drinks a year.  He had smoked significantly up until several months ago.  He now smokes less than three cigarettes a day.,FAMILY HISTORY:, Heart disease in both grandfathers, grandmother with stroke, and a grandmother with diabetes.  Denies obesity and hypertension in other family members.,CURRENT MEDICATIONS:, None.,ALLERGIES:,  He is allergic to Penicillin.,MISCELLANEOUS/EATING HISTORY:, He has been going to support groups for seven months with Lynn Holmberg in Greenwich and he is from Eastchester, New York and he feels that we are the appropriate program.  He had a poor experience with the Greenwich program.  Eating history, he is not an emotional eater.  Does not like sweets.  He likes big portions and carbohydrates.  He likes chicken and not steak.  He currently weighs 312 pounds.  Ideal body weight would be 170 pounds.  He is 142 pounds overweight.  If ,he lost 60% of his excess body weight that would be 84 pounds and he should weigh about 228.,REVIEW OF SYSTEMS: ,Negative for head, neck, heart, lungs, GI, GU, orthopedic, and skin.  Specifically denies chest pain, heart attack, coronary artery disease, congestive heart failure, arrhythmia, atrial fibrillation, pacemaker, high cholesterol, pulmonary embolism, high blood pressure, CVA, venous insufficiency, thrombophlebitis, asthma, shortness of breath, COPD, emphysema, sleep apnea, diabetes, leg and foot swelling, osteoarthritis, rheumatoid arthritis, hiatal hernia, peptic ulcer disease, gallstones, infected gallbladder, pancreatitis, fatty liver, hepatitis, hemorrhoids, rectal bleeding, polyps, incontinence of stool, urinary stress incontinence, or cancer.  Denies cellulitis, pseudotumor cerebri, meningitis, or encephalitis.,PHYSICAL EXAMINATION:, He is alert and oriented x 3.  Cranial nerves II-XII are intact.  Afebrile.  Vital Signs are stable.")

    # List the entities and relations the LLM should look for in the text
    entities = ["Patient", "Medical_Section", "Section_Item"] #PAST_MEDICAL_HISTORY", "PAST_SURGICAL_HISTORY", "SOCIAL_HISTORY", "FAMILY_HISTORY", "CURRENT_MEDICATIONS", "ALLERGIES", "MISCELLANEOUS/EATING_HISTORY", "REVIEW_OF_SYSTEMS", "PHYSICAL_EXAMINATION"]  
    relations = ["HAS_MEDICAL_SECTION","HAS_ITEM"]
    potential_schema = [
        ("Patient", "HAS_MEDICAL_SECTION", "Medical_Section"),
        ("Medical_Section", "HAS_ITEM", "Section_Item"),
    ]

    try:
        # Connect to the Neo4j database
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

        delete_all_graphs(driver)
        delete_index(driver, VECTOR_INDEX_NAME)
        delete_index(driver, FULLTEXT_INDEX_NAME)

        # Create an Embedder object
        embedder = OpenAIEmbeddings(model="text-embedding-3-large")

        #driver, text, entities, relations, potential_schema, embedder,llm
        build_graph(driver, text, entities, relations, potential_schema, embedder, llm)

        # # Generate and store embeddings for the text
        # # Ensure this model generates 1536-dimensional embeddings
        # embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002") 
        # LABELS = ["Patient", "MedicalRecord"]

        # #Imported from neo4j_graphrag.indexes
        # create_vector_index(
        #     driver,
        #     VECTOR_INDEX_NAME,
        #     label=LABELS, #The node label(s) to be indexed.
        #     embedding_property="embedding",
        #     dimensions=3072,
        #     similarity_fn="euclidean",
        # )

        # # Define the fulltext index name, labels, and properties
        # FULLTEXT_PROPERTIES = ["name", "medical_history"]
        # #Imported from neo4j_graphrag.indexes
        # # Create the fulltext index
        # create_fulltext_index(driver, FULLTEXT_INDEX_NAME, LABELS, FULLTEXT_PROPERTIES)

        # generate_and_store_embeddings(embedding_model, driver, text, LABELS)

        # # Initialize the retriever
        # retriever = VectorRetriever(driver, VECTOR_INDEX_NAME, embedder)

        # # Instantiate the RAG pipeline
        # rag = GraphRAG(retriever=retriever, llm=llm)

        # # Query the graph
        # # query_text = "Who is Paul Atreides?"
        # query_text = "What is the medical history of the patient?"
        # response = rag.search(query_text=query_text, retriever_config={"top_k": 5})
        # print(response.answer)
        driver.close()

        # # Example usage of GraphRAG
        # retriever = VectorRetriever(driver, index_name=VECTOR_INDEX_NAME)
        # # Instantiate the LLM
        # # llm = OpenAILLM(model="text-davinci-003")
        # llm = OpenAILLM(
        #     model_name="gpt-4o",
        #     model_params={
        #         "max_tokens": 2000,
        #         "response_format": {"type": "json_object"},
        #         "temperature": 0,
        #     },
        # )
        # rag = GraphRAG(retriever=retriever, llm=llm)

        # # Query the graph
        # query_text = "What is the medical history of the patient?"
        # response = rag.search(query_text=query_text, retriever_config={"top_k": 5})
        # print(response.answer)

        # driver.close()
    except Exception as e:
        print(e)
        driver.close()
        raise e
    finally:
        driver.close()

