import asyncio
import re
import os
import neo4j
import logging
from neo4j_graphrag.embeddings import OpenAIEmbeddings,SentenceTransformerEmbeddings
import neo4j_graphrag
from langchain_ollama import ChatOllama
# from langchain_openai import OpenAIEmbeddings, SentenceTransformerEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.llm.openai_llm import OpenAILLM
from neo4j import GraphDatabase
# from neo4j_graphrag.experimental.components.schema import (
#     SchemaBuilder,
#     SchemaEntity,
#     SchemaRelation,
# )
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.indexes import create_fulltext_index, create_vector_index
from neo4j_graphrag.retrievers import VectorRetriever, HybridRetriever
from neo4j_graphrag.generation import GraphRAG
from langchain_community.vectorstores import Neo4jVector
from neo4j_graphrag.retrievers import HybridCypherRetriever
from langchain_community.graphs import Neo4jGraph
from langchain_ollama import OllamaEmbeddings
from neo4j_graphrag.embeddings.base import Embedder

async def define_and_run_pipeline(
    neo4j_driver: neo4j.Driver,
    llm: LLMInterface,
) -> PipelineResult:
    # Create an instance of the SimpleKGPipeline
    # embedder = neo4j_graphrag.embeddings.OpenAIEmbeddings(model="text-embedding-3-large")
    api_key = "ollama"

    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=neo4j_driver,
        text_splitter=FixedSizeSplitter(chunk_size=4000, chunk_overlap=100),
        embedder=neo4j_graphrag.embeddings.OpenAIEmbeddings(
        base_url="http://localhost:11434/v1",
        api_key=api_key,
        model="llama3",
        ),
        entities=ENTITIES,
        relations=RELATIONS,
        potential_schema=POTENTIAL_SCHEMA,
        from_pdf=False,
        prompt_template=prompt_template,
        perform_entity_resolution = True, #combine entities with similar names
    )
    return await kg_builder.run_async(text=TEXT)

async def build_graph() -> PipelineResult:
    #gpt-4o, gpt-3.5-turbo
    # llm = OpenAILLM(
    #     model_name="gpt-4-turbo",
    #     model_params={
    #         "max_tokens": 4096,
    #         "response_format": {"type": "json_object"},
    #         "temperature": 0.0,
    #         "top_p": 1.0,
    #         "seed": 1313,
    #     },
    # )
    api_key = "ollama"
    print("Connecting to LLM")
    llm = OpenAILLM(
        base_url="http://localhost:11434/v1",
        model_name="llama3",
        api_key=api_key,
        model_params={
            "max_tokens": 20000,
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 1313,
        },
    )
    # with neo4j.GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH, database=NEO4J_DATABASE) as driver:
    res = await define_and_run_pipeline(driver, llm)
    await llm.async_client.close()
    # driver.close()

    return res

def delete_all_graphs():
    # driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH, database=NEO4J_DATABASE)
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    session.close()
    # driver.close()

def list_indexes():
    # driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH, database=NEO4J_DATABASE)
    with driver.session() as session:
        result = session.run("SHOW INDEXES")
        indexes = [record for record in result]
        return indexes
    session.close()
    # driver.close()

def delete_index(index_name):
    # driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH, database=NEO4J_DATABASE)
    # List all indexes
    indexes = list_indexes()
    #checkf if the index exists
    index_exists = False
    for index in indexes:
        if index_name in index:
            index_exists = True
            break

    if index_exists:
        # Delete the vector index
        with driver.session() as session:
            session.run(f"DROP INDEX {index_name}")
        print("Index deleted."+index_name)
    else:
        print("Index does not exist."+index_name)

    # driver.close()

def get_index(index_name):
    # List all indexes
    indexes = list_indexes()
    #checkf if the index exists
    for index in indexes:
        if index_name == index['name']:
            return index

        print("Index does not exist."+index_name)

    return None

def preprocess_text(text):
    #create a pattern to detect SECTION_NAMES which are phrases in all caps followed by a colon
    #Then insert _ to replace spaces
    pattern = r"([A-Z\s]+):"
    text = re.sub(pattern, lambda m: m.group(1).replace(" ", "_") + ":", text)
    return text

def do_rag(llm, retriever, candidate):

    questions = """
    1. Provide a summary of the health status of the patient {CANDIDATE}
    2. What is the past medical history of patient {CANDIDATE}?
    3. What is the past surgical history of patient {CANDIDATE}?
    4. What is the social history of patient {CANDIDATE}?
    5. What is the family history of patient {CANDIDATE}?
    6. What are the current medications of patient {CANDIDATE}?
    7. What are the allergies of patient {CANDIDATE}?
    8. What is the eating history of patient {CANDIDATE}?
    9. What is the review of systems for patient {CANDIDATE}?
    10. What are the physical examination details for patient {CANDIDATE}?
    """

    QUESTIONS = questions.split("\n")
    #remove empty strings from the list of questions
    QUESTIONS = [q.strip() for q in QUESTIONS if q.strip()]

    summary_of_answers = ""
    question_number = 0

    # Format the prompt with the tools and context
    # formatted_prompt = prompt_template.format(tools="\n".join(tool_names), context=result)

    for q in QUESTIONS:
        user_query = q.replace("{CANDIDATE}", candidate)

        rag = GraphRAG(retriever=retriever, llm=llm)
        response = rag.search(query_text=user_query, retriever_config={"top_k": 5})
        
        answer = response.answer
        summary_of_answers += "\nQuestion: " + user_query + "\n"
        #result_1['source_documents'] returns a list of references[0]['text']
        summary_of_answers += f"Answer: " + answer + "\n"
        # chat_history.append(HumanMessage(content=user_query))
        # chat_history.append(AIMessage(content=answer))

        question_number+=1

    print("RAG Answers")
    print(summary_of_answers)

#Apparently returning and capturing the vector index results in a neo4j memory/resource leak with this function
def generate_vector_index(index_name, node_label, text_node_properties, embedding_node_property, embedding, url,auth):

    vector_index = Neo4jVector.from_existing_graph(
        embedding=embedding,
        url=url,
        username=auth[0],
        password=auth[1],
        index_name=index_name,
        node_label=node_label,
        text_node_properties=text_node_properties,
        embedding_node_property=embedding_node_property,
    )
    # # Connect to the Neo4j database
    # driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH, database=NEO4J_DATABASE)

    # # Create the index
    # create_vector_index(
    #     driver,
    #     index_name,
    #     label="Chunk",
    #     embedding_property=embedding_node_property,
    #     dimensions=3072,
    #     similarity_fn="euclidean",
    # )
    # driver.close()
    
if __name__ == "__main__":

    # Neo4j db infos
    NEO4J_URI = "neo4j://localhost:7687"
    NEO4J_AUTH = ("neo4j", "MyNeo4J@2024")
    NEO4J_DATABASE = "neo4j"

    # Define the name of the index
    VECTOR_INDEX_NAME = "medical_vector_index"
    FULLTEXT_INDEX_NAME = "medical_fulltext_index"
    
    DIMENSION = 1536 #Used for vector embeddings

    # TEXT = ("Medical transcription for Patient_P001 is given below."
        # "PAST MEDICAL HISTORY:, He has difficulty climbing stairs, difficulty with airline seats, tying shoes, used to public seating, and lifting objects off the floor.  He exercises three times a week at home and does cardio.  He has difficulty walking two blocks or five flights of stairs.  Difficulty with snoring.  He has muscle and joint pains including knee pain, back pain, foot and ankle pain, and swelling.  He has gastroesophageal reflux disease.,PAST SURGICAL HISTORY:, Includes reconstructive surgery on his right hand thirteen years ago.  ,SOCIAL HISTORY:, He is currently single.  He has about ten drinks a year.  He had smoked significantly up until several months ago.  He now smokes less than three cigarettes a day.,FAMILY HISTORY:, Heart disease in both grandfathers, grandmother with stroke, and a grandmother with diabetes.  Denies obesity and hypertension in other family members.,CURRENT MEDICATIONS:, None.,ALLERGIES:,  He is allergic to Penicillin.,MISCELLANEOUS/EATING HISTORY:, He has been going to support groups for seven months with Lynn Holmberg in Greenwich and he is from Eastchester, New York and he feels that we are the appropriate program.  He had a poor experience with the Greenwich program.  Eating history, he is not an emotional eater.  Does not like sweets.  He likes big portions and carbohydrates.  He likes chicken and not steak.  He currently weighs 312 pounds.  Ideal body weight would be 170 pounds.  He is 142 pounds overweight.  If ,he lost 60% of his excess body weight that would be 84 pounds and he should weigh about 228.,REVIEW OF SYSTEMS: ,Negative for head, neck, heart, lungs, GI, GU, orthopedic, and skin.  Specifically denies chest pain, heart attack, coronary artery disease, congestive heart failure, arrhythmia, atrial fibrillation, pacemaker, high cholesterol, pulmonary embolism, high blood pressure, CVA, venous insufficiency, thrombophlebitis, asthma, shortness of breath, COPD, emphysema, sleep apnea, diabetes, leg and foot swelling, osteoarthritis, rheumatoid arthritis, hiatal hernia, peptic ulcer disease, gallstones, infected gallbladder, pancreatitis, fatty liver, hepatitis, hemorrhoids, rectal bleeding, polyps, incontinence of stool, urinary stress incontinence, or cancer.  Denies cellulitis, pseudotumor cerebri, meningitis, or encephalitis.,PHYSICAL EXAMINATION:, He is alert and oriented x 3.  Cranial nerves II-XII are intact.  Afebrile.  Vital Signs are stable.")
    TEXT = ("Medical transcription for Patient_ABC is given below."
            "HISTORY OF PRESENT ILLNESS: , I have seen ABC today.  He is a very pleasant gentleman who is 42 years old, 344 pounds.  He is 5'9\".  He has a BMI of 51.  He has been overweight for ten years since the age of 33, at his highest he was 358 pounds, at his lowest 260.  He is pursuing surgical attempts of weight loss to feel good, get healthy, and begin to exercise again.  He wants to be able to exercise and play volleyball.  Physically, he is sluggish.  He gets tired quickly.  He does not go out often.  When he loses weight he always regains it and he gains back more than he lost.  His biggest weight loss is 25 pounds and it was three months before he gained it back.  He did six months of not drinking alcohol and not taking in many calories.  He has been on multiple commercial weight loss programs including Slim Fast for one month one year ago and Atkin's Diet for one month two years ago.,PAST MEDICAL HISTORY: , He has difficulty climbing stairs, difficulty with airline seats, tying shoes, used to public seating, difficulty walking, high cholesterol, and high blood pressure.  He has asthma and difficulty walking two blocks or going eight to ten steps.  He has sleep apnea and snoring.  He is a diabetic, on medication.  He has joint pain, knee pain, back pain, foot and ankle pain, leg and foot swelling.  He has hemorrhoids.,PAST SURGICAL HISTORY: , Includes orthopedic or knee surgery.,SOCIAL HISTORY: , He is currently single.  He drinks alcohol ten to twelve drinks a week, but does not drink five days a week and then will binge drink.  He smokes one and a half pack a day for 15 years, but he has recently stopped smoking for the past two weeks.,FAMILY HISTORY: , Obesity, heart disease, and diabetes.  Family history is negative for hypertension and stroke.,CURRENT MEDICATIONS:,  Include Diovan, Crestor, and Tricor.,MISCELLANEOUS/EATING HISTORY:  ,He says a couple of friends of his have had heart attacks and have had died.  He used to drink everyday, but stopped two years ago.  He now only drinks on weekends.  He is on his second week of Chantix, which is a medication to come off smoking completely.  Eating, he eats bad food.  He is single.  He eats things like bacon, eggs, and cheese, cheeseburgers, fast food, eats four times a day, seven in the morning, at noon, 9 p.m., and 2 a.m.  He currently weighs 344 pounds and 5'9\".  His ideal body weight is 160 pounds.  He is 184 pounds overweight.  If he lost 70% of his excess body weight that would be 129 pounds and that would get him down to 215.,REVIEW OF SYSTEMS: , Negative for head, neck, heart, lungs, GI, GU, orthopedic, or skin.  He also is positive for gout.  He denies chest pain, heart attack, coronary artery disease, congestive heart failure, arrhythmia, atrial fibrillation, pacemaker, pulmonary embolism, or CVA.  He denies venous insufficiency or thrombophlebitis.  Denies shortness of breath, COPD, or emphysema.  Denies thyroid problems, hip pain, osteoarthritis, rheumatoid arthritis, GERD, hiatal hernia, peptic ulcer disease, gallstones, infected gallbladder, pancreatitis, fatty liver, hepatitis, rectal bleeding, polyps, incontinence of stool, urinary stress incontinence, or cancer.  He denies cellulitis, pseudotumor cerebri, meningitis, or encephalitis.,PHYSICAL EXAMINATION:  ,He is alert and oriented x 3.  Cranial nerves II-XII are intact.  Neck is soft and supple.  Lungs:  He has positive wheezing bilaterally.  Heart is regular rhythm and rate.  His abdomen is soft.  Extremities:  He has 1+ pitting edema.,IMPRESSION/PLAN:,  I have explained to him the risks and potential complications of laparoscopic gastric bypass in detail and these include bleeding, infection, deep venous thrombosis, pulmonary embolism, leakage from the gastrojejuno-anastomosis, jejunojejuno-anastomosis, and possible bowel obstruction among other potential complications.  He understands.  He wants to proceed with workup and evaluation for laparoscopic Roux-en-Y gastric bypass.  He will need to get a letter of approval from Dr. XYZ.  He will need to see a nutritionist and mental health worker.  He will need an upper endoscopy by either Dr. XYZ.  He will need to go to Dr. XYZ as he previously had a sleep study.  We will need another sleep study.  He will need H. pylori testing, thyroid function tests, LFTs, glycosylated hemoglobin, and fasting blood sugar.  After this is performed, we will submit him for insurance approval.")

    # List the entities and relations the LLM should look for in the text
    ENTITIES = [
        "PATIENT", 
        "MEDICAL_SECTION",
        "SECTION_ITEM"]  
    RELATIONS = [
        "HAS_MEDICAL_SECTION",
        "HAS_ITEM"]
    POTENTIAL_SCHEMA = [
        ("PATIENT", "HAS_MEDICAL_SECTION", "MEDICAL_SECTION"),
        ("MEDICAL_SECTION", "HAS_ITEM", "SECTION_ITEM"),
    ]
    prompt_template = '''
    You are a helpful agent designed to construct a graph from text data containing patient transcriptions. 

    A patient transcription consists of the medical sections like: 
    Past Medical History, Surgical History, Social History, Family History, Current Medications,
    Allergies, Miscellaneous/Eating History, Review of Systems, Physical Examination etc.

    Each medical section contains:
    1. A section name in all caps, followed by a semicolon and sometimes a comma.
    2. Addional comma-separated information about the patient's health, pertaining to that section.

    The main Patient entity is the root node of the graph and is connected to the Medical Sections via the HAS_MEDICAL_SECTION relationship.
    The Patient entity should have a name, in all uppercase, in the format: PATIENT_P<ID>, where ID is a number.

    You need to extract the section names and the information contained in each section. The section names are the entities of type MEDICAL_SECTION
    and the information contained in each section are also entities of type SECTION_ITEM. Each SECTION_ITEM entity should have a name in
    the format: SECTION_ITEM_<nnn>, where nnn is a sequence number and the text associated with the section item should be 
    stored in a property called section_item_text. The text stored in the section_item_text property of each SECTION_ITEM entity
    should be prefixed with the section name and a colon

    Extract the entities (nodes) and specify their type from the following Input text.
    Also extract the relationships between these nodes. the relationship direction goes from the start node to the end node. 

    Return result as JSON using the following format:
    {{"nodes": [ {{"id": "0", "label": "the type of entity", "properties": {{"name": "name of entity" }} }}],
    "relationships": [{{"type": "TYPE_OF_RELATIONSHIP", "start_node_id": "0", "end_node_id": "1", "properties": {{"details": "Description of the relationship"}} }}] }}

    - Use only the information from the Input text. Do not add any additional information.  
    - If the input text is empty, return empty Json. 
    - Make sure to create as many nodes and relationships as needed to offer rich medical context for further research.
    - An AI knowledge assistant must be able to read this graph and immediately understand the context to inform detailed research questions. 
    - Multiple documents will be ingested from different sources and we are using this property graph to connect information, so make sure entity types are fairly general. 
    - Every Medical Section entity should be connected to the Patient entity with a HAS_MEDICAL_SECTION relationship.

    Use only fhe following nodes and relationships (if provided):
    {schema}

    Assign a unique ID (string) to each node, and reuse it to define relationships.
    Do respect the source and target node types for relationship and
    the relationship direction.

    Do not return any additional information other than the JSON in it.
    Use the examples below only as a reference and do not include them in the output.

    Examples:
    <EXAMPLE_START>
    ["Patient_001"->"HAS_MEDICAL_SECTION"->"PAST_MEDICAL_HISTORY"]
    ["Patient_001"->"HAS_MEDICAL_SECTION"->"PAST_SURGICAL_HISTORY"]
    ["PAST_MEDICAL_HISTORY"->"HAS_ITEM"->"SECTION_ITEM_001"]
    ["PAST_MEDICAL_HISTORY"->"HAS_ITEM"->"SECTION_ITEM_002"]
    ["PAST_SURGICAL_HISTORY"->"HAS_ITEM"->"SECTION_ITEM_001"]
    <EXAMPLE_END>

    Input text:
    {text}
    '''
    
    try:
        
        # Connect to the Neo4j database
        driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH,database=NEO4J_DATABASE)

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        # Disable Neo4j warnings
        logging.getLogger("neo4j").setLevel(logging.INFO)

        useOllama = False
        if useOllama:
            # Instantiate the LLM        
            llm = ChatOllama(model='llama3', config={'max_new_tokens': 20000, 'temperature': 0.0, 'top_p':1.0, 'seed':42, 'context_length': 10000})
            # Create an Embedder object
            embedder = OllamaEmbeddings(model="mxbai-embed-large")
        else:
            #check if OPENAI_API_KEY is set
            if "OPENAI_API_KEY" not in os.environ:
                print("OPENAI_API_KEY environment variable is not set.")
                raise Exception("OPENAI_API_KEY environment variable is not set.")
            else:
                #Check if the API key is valid
                api_key = os.environ["OPENAI_API_KEY"]
                #api_key should ne atleast 32 characters long
                if len(api_key) < 32:
                    print("OPENAI_API_KEY environment variable is not set.")
                    raise Exception("OPENAI_API_KEY environment variable is not set or is invalid.")
            # Instantiate the LLM        
            # "gpt-3.5-turbo",gpt-4-turbo
            llm = OpenAILLM(
                model_name="gpt-3.5-turbo",
                model_params={
                    "max_tokens": 4096,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "seed": 42,
                },
            )
            # Create an Embedder object
            embedder = OpenAIEmbeddings(model="text-embedding-3-small")

        rebuild = True
        if rebuild:
            TEXT = preprocess_text(TEXT)
            # # if rebuild:
            # '''
            #indexes have to be cleared properly for LLM rag to work
            delete_all_graphs()
            delete_index(VECTOR_INDEX_NAME)
            delete_index(FULLTEXT_INDEX_NAME)

            res = asyncio.run(build_graph())
            print(res)

            LABEL = "SECTION_ITEM"
            property_name = "section_item_text"
            vector_property_name = "vectorProperty"

            # '''
            generate_vector_index(VECTOR_INDEX_NAME, LABEL, [property_name], vector_property_name, embedder ,NEO4J_URI,NEO4J_AUTH)  
            print("Index created."+VECTOR_INDEX_NAME)
            #     # '''

            # with neo4j.GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH, database=NEO4J_DATABASE) as driver:
            # Imported from neo4j_graphrag.indexes
            create_fulltext_index(
                driver, FULLTEXT_INDEX_NAME, label=LABEL, node_properties=[vector_property_name]
            )
            print("Index created."+FULLTEXT_INDEX_NAME)
            # '''
        else:
            # Query the graph
            # user_query = "What is the past surgical history of the patient?"
            # context_documents = vector_index.similarity_search(user_query, k=5)
            #The response is a list of dictionaries containing the node and the similarity score
            #The node is a dictionary containing the node properties
            #Iterate through the response to get the node property section_item_text and the similarity score
            # initial_context = "\n".join([doc.page_content for doc in context_documents])
            # print("Initial context:", initial_context)

            # Simple vector retriever that uses vector similarity search to retrieve nodes from the knowledge graph
            # retriever = VectorRetriever(driver, index_name=VECTOR_INDEX_NAME, embedder=embedder)

            # HybridRetriever uses both a vector index and a full-text index to carry out a hybrid search.
            # It uses the user query to search both indexes, retrieving nodes and their corresponding scores.
            # After normalizing the scores from each set of results, it merges them, ranks the combined results by 
            # score, and returns the top matches.
            # Hybrid retriever results in a very large number of tokens being sent to the LLM and 
            # this results in an error response from OPenAI due to token limits
            # retriever = HybridRetriever(driver, VECTOR_INDEX_NAME, FULLTEXT_INDEX_NAME, embedder)

            # VectorCypherRetriever first retrieves an initial series of nodes from the knowledge graph 
            # using vector search, then uses a Cypher query to traverse the graph from each of these initial 
            # nodes and gather the additional information from the nodes connected to them.
            # Hybrid retriever results in a very large number of tokens being sent to the LLM and 
            # this results in an error response from OPenAI due to token limits
        
            #HybridCypherSearch
            RETRIEVAL_QUERY = """
            MATCH (n:MEDICAL_SECTION)-[:HAS_ITEM]->(p:SECTION_ITEM) 
            RETURN n.name as medicalSectionName, collect(p.name) as medicalSectionItemNames,
            collect(p.section_item_text) as medicalSectionItemTexts,
            score as similarityScore"""

            retriever = HybridCypherRetriever(
                driver=driver,
                vector_index_name=VECTOR_INDEX_NAME,
                fulltext_index_name=FULLTEXT_INDEX_NAME,
                # note: embedder is optional if you only use query_vector
                embedder=embedder,
                retrieval_query=RETRIEVAL_QUERY,
                # optionally, configure how to format the results
                # (see corresponding example in 'customize' directory)
                # result_formatter=None,
                # optionally, set neo4j database
                # neo4j_database="neo4j",
            )

            candidate = "Patient_ABC"
            print ("Running RAG for candidate:", candidate)
            do_rag(llm, retriever, candidate)

    except Exception as e:
        print(f"An error occurred: {e}")
        logger.error("An error occurred: %s", e)
    finally:
        if driver is not None:
            driver.close()
            logger.debug("Database connection closed.")
    