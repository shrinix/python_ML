from langchain.prompts import PromptTemplate
from langchain.llms.openai import OpenAI
# from langchain.llms.ollama import Ollama
from langchain.chains import LLMChain
from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.networkx_graph import KG_TRIPLE_DELIMITER
from pprint import pprint
from pyvis.network import Network
import networkx as nx
import gradio as gr
import os
from graph import GraphBuilder
from neo4j import GraphDatabase
import py2neo
from py2neo import Graph, Node, Relationship
import pyvis.network
import re
import neo4j
# from langchain_community.document_loaders import TextLoader
# from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import NLTKTextSplitter
import pyvis
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
import webbrowser
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import StringPromptTemplate

# import nltk
# import ssl
# Disable SSL certificate verification
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# # Download the required NLTK data
# nltk.download('punkt_tab')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "MyNeo4J@2024"

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Prompt template for knowledge triple extraction
_DEFAULT_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE = (
    "You are a networked intelligence helping a human track knowledge triples"
    " about all relevant people, things, concepts, etc. and integrating"
    " them with your knowledge stored within your weights"
    " as well as that stored in a knowledge graph."
    " Extract all of the knowledge triples from the text."
    " A knowledge triple is a clause that contains a subject, a predicate,"
    " and an object. The subject is the entity being described,"
    " the predicate is the property of the subject that is being"
    " described, and the object is the value of the property."
    "The subject and object are entities, and the predicate is a relationship."
    "The subject is usually a noun, the predicate is a verb, and the object is a noun or a pronoun."
    "Ensure that you follow the below instructions while extracting the knowledge triples:"
    "1. Instructions for handling Numerical Data and Dates:"
    "- Numerical data, like age or other related information, should be incorporated as attributes or properties of the respective nodes."
    # "- **No Separate Nodes for Dates/Numbers**: Do not create separate nodes for dates or numerical values. Always attach them as attributes or properties of nodes."
    "- **Property Format**: Properties must be in a key-value format."
    "- **Quotation Marks**: Never use escaped single or double quotes within property values."
    # "- **Naming Convention**: Use camelCase for property keys, e.g., `birthDate`."
    "- **Numerical or Date data cannot be used as a subject or object."
    "- **Subjects and Objects cannot start with a number or a date. For example, \"13 years ago\" should be part of the predicate, not the subject or the object."
    # "2. Coreference Resolution:"
    # "- **Maintain Entity Consistency**: When extracting entities, it's vital to ensure consistency."
    # "If an entity, such as \"John Doe\", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., \"Joe\", \"he\"),"
    # "always use the most complete identifier for that entity throughout the knowledge graph. In this example, use \"John Doe\" as the entity ID."
    # "Remember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial."
    "Some examples of triple construction from sentences are given below for reference. Do not use these examples as input to the model. Do not use the examples as a template for the output."
    "\n\nEXAMPLE\n"
    "It's a state in the US. It's also the number 1 producer of gold in the US.\n\n"
    f"Output: (Nevada, is a, state){KG_TRIPLE_DELIMITER}(Nevada, is in, US)"
    f"{KG_TRIPLE_DELIMITER}(Nevada, is the number 1 producer of, gold)\n"
    "END OF EXAMPLE\n\n"
    "EXAMPLE\n"
    "I'm going to the store.\n\n"
    "Output: NONE\n"
    "END OF EXAMPLE\n\n"
    "EXAMPLE\n"
    "Oh huh. I know Descartes likes to drive antique scooters and play the mandolin.\n"
    f"Output: (Descartes, likes to drive, antique scooters){KG_TRIPLE_DELIMITER}(Descartes, plays, mandolin)\n"
    "END OF EXAMPLE\n\n"
    "{text}"
    "Output:"
)

def preprocess_transcription_data(tr_text):

    #Remove words or phrases that are all-caps and end with the character :
    tr_text = re.sub(r'\b[A-Z\s]+\b:', '', tr_text)

    # #getting rid of targeted charachters in the trascription
    # chars = ['#',':,',': ,',';','$','!','?','*','``','1. ', '2. ', '3. ', '4. ', '5. ','6. ','7. ','8. ','9. ','10. ']
    # for c in chars:
    #     tr_text = tr_text.replace(c,"")

    # #getting rid of targeted charachters in the trascription
    # chars = [",", ".", "[", "]", ":", "``", ")", "(", "1", "2", "5", "%", "3", "4", "4-0", "3-0", "6", "''", "0", "2-0", "8", "7", "&", "5-0", "9", "0.5", "1.5", "500", "50", "100", "6-0", "15", "2.5", "14-15", "60", "'", "300", "14", "________", "7-0", "90", "__________", "3.5", "1:100,000", "70", "0.", "80", "1:50,000", "03/08/200 ", "03/09/2007", "25605", "7.314", "33.0", "855.", "08/22/03", "10/500", "125.", "144/6"]
    # for c in chars:
    #     tr_text = tr_text.replace(c," ")

    return tr_text

def parse_triples(response, delimiter=KG_TRIPLE_DELIMITER):
    if not response:
        return []
    return response.split(delimiter)

#function to #construct a templatized cypher query to create a node for the subject and object and a relationship between them
def construct_cypher_query(patient_name, graph_name, section_name, triplet):
    #triplet is a string of the form '(subject, predicate, object)'
    #split the string into a list of three elements
    triplet = triplet.strip()
    triplet = triplet[1:-1]
    #extract 3 comma separated values from the triplet
    triplet = triplet.split(',')
    subject, predicate, object = triplet
    #remove any leading or trailing whitespaces from the subject, object and predicate
    subject = subject.strip()
    object = object.strip()
    predicate = predicate.strip()
    #if the subject is a multi-word entity, replace the spaces with underscores
    subject = subject.replace(' ', '_')
    subject = subject.replace('-', '_')
    subject = subject.replace('/', '_')
    object = object.replace(' ', '_')
    object = object.replace('-', '_')
    object = object.replace('/', '_')

    # Construct the Cypher query
    query = f"""
        MATCH (e:Entity {{name: '{section_name}', type: 'Medical_Section', graph: '{graph_name}'}})
        MERGE (s:Entity {{name: '{subject}', type: '{subject}', graph: '{graph_name}'}})
        MERGE (e)-[:MENTIONS]->(s)
        MERGE (o:Entity {{name: '{object}', type: '{object}', graph: '{graph_name}'}})
        MERGE (s)-[:`{predicate}` {{graph: '{graph_name}'}}]->(o)
    """
    
    return query

#function to create a graph from the extracted triples. Use Neo4jGraph to create the graph
def create_neo4j_graph_from_triplets(patient_name, graph_name, section_name, triples_list):

    # Connect to Neo4j
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "MyNeo4J@2024"))

    # Create nodes and relationships
    with driver.session() as session:

        # Create a graph with the specified name
        session.run(f"CREATE (:Graph {{name: '{graph_name}'}})")
        for triplet in triples_list:
            try:
                query = construct_cypher_query(patient_name, graph_name, section_name, triplet)
                session.run(query)
            except ValueError:
                print(f"Skipping invalid triplet: {triplet}")

    # Close the driver
    driver.close()

def get_all_nodes_and_relationships():
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "MyNeo4J@2024"))
                                                                
    # try:
    #     with driver.session() as session:
    #         result = session.run("MATCH (n) RETURN n")
    #         return [record["n"] for record in result]
    # finally:
    #     driver.close()

        # Query to get a graphy result
    graph_result = driver.execute_query("""
        MATCH (n)-[r]->(m) RETURN n, r, m
        """,
        result_transformer_=neo4j.Result.graph,
    )

    return graph_result

def create_triples_dict(patient_name, text):

    # Run the chain with the specified text
    #The text has multiple subsections each of which has a format: <SECTION_NAME>:, <SECTION_CONTENT>
    #Get the individual sections by splitting the text based on the pattern: <SECTION_NAME>:
    #pattern to recognize the start of a new section
    pattern = re.compile(r'[A-Z\s/]+:\s?,?\s?')
    #trim the text to remove any leading or trailing whitespaces
    text = text.strip()
    # Find all matches of the pattern in the text
    section_names = pattern.findall(text)
    # Clean the section names by removing the trailing colon and comma
    section_names = [name[:-3].strip() for name in section_names]
    # Print the list of section names
    print(section_names)

    # Split the text into sections based on the pattern
    sections = pattern.split(text)

    # Remove the first split item if it is a blank line
    if sections[0] == '':
        sections = sections[1:]

    #Replace any special characters in the section names with underscores
    section_names = [re.sub(r'\W+', '_', name) for name in section_names]

    # Remove any empty strings
    sections = [s for s in sections if s]

    # Print the sections for debugging
    for i, section in enumerate(sections):
        print(f"Section {i + 1}: {section}")

    # Build the dictionary
    section_dict = {name: content.strip() for name, content in zip(section_names, sections)}

    #Change the key names in the dictionary to include the patient name
    section_dict = {f"{key}": value for key, value in section_dict.items()}

    #create a new dictionary to store the patient name with the section names
    patient_sections_dict = {patient_name: section_names}

    # Print the dictionary for debugging
    print("Section Dictionary:")
    for key, value in section_dict.items():
        print(f"{key}: {value}")

    #if any section has less than 10 characters, remove it
    section_dict = {key: value for key, value in section_dict.items() if len(value) > 10}

    # text = preprocess_transcription_data(text)

    # loader = TextLoader(text=text)
    # documents = loader.load()
    # text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=20)
    # texts = text_splitter.split_documents(documents)

    # splitter = NLTKTextSplitter(chunk_size=50,chunk_overlap=20)
    # chunks = splitter.split_text(text)

    for key, value in section_dict.items():
        # Run the chain with the specified text
        triples = chain.invoke(
            {'text' : value}
        ).get('text')

        triples_list = parse_triples(triples)

        # From the extracted triples, remove any items which are not triples i.e. they have only one comma
        triples_list = [t for t in triples_list if t.count(',') == 2]
        #trim any leading or trailing whitespaces from the triples
        triples_list = [t.strip() for t in triples_list]
        print(f"Triples for section {key}:")
        pprint(triples_list)

        #create a new tuple to store both the text and the triples for the section
        section_triples_tuple = (value, triples_list)
        section_dict[key] = section_triples_tuple

    return patient_sections_dict, section_dict

def view_graph():
    graph = get_all_nodes_and_relationships()

    # nodes_text_properties = {  # what property to use as text for each node
    #     "Entity": "name",
    # }

    nodes_text_properties = {"Entity": "name", "Graph": "graph"}
    
    html_content = visualize_result(graph, nodes_text_properties)
    return html_content

def visualize_result(query_graph, nodes_text_properties):
    visual_graph = pyvis.network.Network()

    # Extract nodes and relationships from the Graph object
    nodes = query_graph.nodes
    relationships = query_graph.relationships

    # Iterate over the list of nodes
    for node in nodes:
        node_label = list(node.labels)[0]
        node_text = node[nodes_text_properties[node_label]]
        visual_graph.add_node(node.id, node_text, group=node_label)

    # Iterate over the list of relationships
    for relationship in relationships:
        visual_graph.add_edge(
            relationship.start_node.id,
            relationship.end_node.id,
            title=relationship.type
        )

    # visual_graph.show('network.html', notebook=False)
    #     # Open the HTML file in the default web browser
    # file_path = os.path.abspath('network.html')
    # webbrowser.open(f'file://{file_path}')

    html_content = visual_graph.generate_html()
    return html_content

    # html = html.replace("'", "\"")

    # return f"""<iframe style="width: 100%; height: 600px;margin:0 auto" name="result" allow="midi; geolocation; microphone; camera;
    # display-capture; encrypted-media;" sandbox="allow-modals allow-forms
    # allow-scripts allow-same-origin allow-popups
    # allow-top-navigation-by-user-activation allow-downloads" allowfullscreen=""
    # allowpaymentrequest="" frameborder="0" srcdoc='{html}'></iframe>"""

def delete_vector_embeddings_and_vector_index(index_name, node_label, embedding_node_property,text_node_properties):
     # Connect to Neo4j
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "MyNeo4J@2024"))

    # Check if the vector index exists
    index_exists_query = f"SHOW INDEXES WHERE name = '{index_name}'"
    index_exists_result = graph.run(index_exists_query).data()
    index_exists = len(index_exists_result) > 0

    # Check if the embeddings exist
    embeddings_exist_query = f"MATCH (n:{node_label}) WHERE n.{embedding_node_property} IS NOT NULL RETURN count(n) AS count"
    embeddings_exist_result = graph.run(embeddings_exist_query).data()
    embeddings_exist = embeddings_exist_result[0]['count'] > 0

    if index_exists:
        # Delete the vector index
        #TODO: Not correct invocation of the delete method
        vector_index = Neo4jVector.from_existing_graph(
            OpenAIEmbeddings(),
            url=os.environ["NEO4J_URI"],
            username=os.environ["NEO4J_USERNAME"],
            password=os.environ["NEO4J_PASSWORD"],
            index_name=index_name,
            node_label=node_label,
            text_node_properties=text_node_properties,
            embedding_node_property=embedding_node_property,
        )
        vector_index.delete()
        print(f"Index '{index_name}' has been deleted.")
    else:
        print(f"Index '{index_name}' does not exist.")

    if embeddings_exist:
        # Define the Cypher query to remove the vector embeddings
        query = f"""
        MATCH (n:{node_label})
        REMOVE n.{embedding_node_property}
        RETURN count(n) AS nodes_updated
        """

        # Run the query to remove the embeddings
        result = graph.run(query)
        nodes_updated = result.single()["nodes_updated"]
        print(f"Vector embeddings removed from {nodes_updated} nodes.")
    else:
        print(f"No embeddings found for property '{embedding_node_property}' on nodes with label '{node_label}'.")

def create_vector_embeddings_and_vector_index(embeddings_model,index_name, node_label, text_node_properties, embedding_node_property):

    from langchain.vectorstores.neo4j_vector import Neo4jVector
    vector_index = Neo4jVector.from_existing_graph(
        OpenAIEmbeddings(model=embeddings_model),
        url=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
        index_name=index_name,
        node_label=node_label,
        text_node_properties=text_node_properties,
        embedding_node_property=embedding_node_property,
    )
    
    return vector_index

def create_level_1_graph(patient_name, patient_sections_dict, sections_triples_dict):
    #Use the hierarchical structure captured in the dictionary to create a graph.
    #the first level of the graph should be the patient name and the second level should be the section names.
    #for each section, extract the list of triples and create a graph from it. Also add a property to each node to indicate the section it belongs to.  

    # Connect to Neo4j
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "MyNeo4J@2024"))

    # Create nodes and relationships
    with driver.session() as session:

        # Create a graph with the specified name
        graph_name = patient_name+"_Medical_Transcription"
        session.run(f"CREATE (:Graph {{name: '{graph_name}'}})")
        #get the sections for the patient
        section_names = patient_sections_dict[patient_name]

        try:
            # Construct the Cypher query to connect patient nodes to section nodes
            for section_name in section_names:
                section_name = section_name.strip()
                section_name = section_name.replace(' ', '_')
                section_name = section_name.replace('-', '_')
                section_name = section_name.replace('/', '_')

                query = (
                    f"MERGE (p:Entity {{name: '{patient_name}', type: 'Person', graph: '{graph_name}'}}) "
                    f"MERGE (s:Entity {{name: '{section_name}', type: 'Medical_Section', graph: '{graph_name}'}}) "
                    f"MERGE (p)-[:`HAS_SECTION` {{graph: '{graph_name}'}}]->(s)"
                )
                session.run(query)
        except ValueError:
            print(f"Skipping invalid section name: {section_name}")
    # Close the driver
    driver.close()

    return graph_name

def build_new_graph(patient_sections_dict, sections_triples_dict):
    #Use the hierarchical structure captured in the dictionary to create a graph.
    #the first level of the graph should be the patient name and the second level should be the section names.
    #for each section, extract the list of triples and create a graph from it. Also add a property to each node to indicate the section it belongs to.  

    graph_name = create_level_1_graph(patient_name, patient_sections_dict, sections_triples_dict)

    for section_name, section_tuple in sections_triples_dict.items():
        # text = section_tuple[0]
        triples_list = section_tuple[1]
        #create a graph from the extracted triples
        create_neo4j_graph_from_triplets(patient_name, graph_name, section_name, triples_list)
    
def build_graph(graph_builder, dict):

    #Dict is a dictionary that contains entries like: {section name = (section text, list of triples from section text)}
    #for each section, extract the list of triples and create a graph from it.
    for key, value in dict.items():
        text = value[0]
        triples_list = value[1]
        #create a graph from the extracted triples
        create_neo4j_graph_from_triplets(key, triples_list)

def query_db_by_section(section_name):
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "MyNeo4J@2024"))                                                            

    #create a cypher query to get all the nodes and relationships for the specified section
    query = f"""
        MATCH (e:Entity {{name: '{section_name}'}})"""

    # Query to get a graphy result
    result = driver.execute_query(query)
    print("Tool invoked with result:"+str(result))

    return result

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
        
    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        # intermediate_steps = kwargs.pop("intermediate_steps")
        # thoughts = ""
        # for action, observation in intermediate_steps:
        #     thoughts += action.log
        #     thoughts += f"\nObservation: {observation}\nThought: "
        # # Set the agent_scratchpad variable to that value
        # kwargs["agent_scratchpad"] = thoughts
        # ############## NEW ######################
        #tools = self.tools_getter(kwargs["input"])
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
        # kwargs["entity_types"] = json.dumps(entity_types)

        # Ensure the 'query' key is present in kwargs
        if "query" not in kwargs:
            kwargs["query"] = "default query"  # Replace with an appropriate default value or raise an error

        return self.template.format(**kwargs)

def fetch_related_nodes_and_relationships(node_names):
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "MyNeo4J@2024"))
    # Define the Cypher query to fetch related nodes and relationships
    query = f"""
    MATCH (n)-[r]-(m)
    WHERE n.name IN {node_names}
    RETURN n, r, m
    """
    # Run the query
    result = graph.run(query)
    return result

class CustomRetriever():
        def __init__(self, retriever):
            self.retriever = retriever

        def retrieve(self, query):
            context = self.retriever.retrieve(query)
            print(f"Context for query '{query}': {context}")
            return context
        
if __name__ == "__main__":

    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, top_p=1.0, seed=42)
    llm = ChatOllama(model='mistral', config={'max_new_tokens': 1024, 'temperature': 0.0, 'top_p':1.0, 'seed':42, 'context_length': 25000})
    KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT = PromptTemplate(
        input_variables=["text"],
        template=_DEFAULT_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE,
    )

    # Create an LLMChain using the knowledge triple extraction prompt
    chain = LLMChain(llm=llm, prompt=KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT)

    # text = "SUBJECTIVE:,  This 23-year-old white female presents with complaint of allergies.  She used to have allergies when she lived in Seattle but she thinks they are worse here.  In the past, she has tried Claritin, and Zyrtec.  Both worked for short time but then seemed to lose effectiveness.  She has used Allegra also.  She used that last summer and she began using it again two weeks ago.  It does not appear to be working very well.  She has used over-the-counter sprays but no prescription nasal sprays.  She does have asthma but doest not require daily medication for this and does not think it is flaring up.,MEDICATIONS: , Her only medication currently is Ortho Tri-Cyclen and the Allegra.,ALLERGIES: , She has no known medicine allergies.,OBJECTIVE:,Vitals:  Weight was 130 pounds and blood pressure 124/78.,HEENT:  Her throat was mildly erythematous without exudate.  Nasal mucosa was erythematous and swollen.  Only clear drainage was seen.  TMs were clear.,Neck:  Supple without adenopathy.,Lungs:  Clear.,ASSESSMENT:,  Allergic rhinitis.,PLAN:,1.  She will try Zyrtec instead of Allegra again.  Another option will be to use loratadine.  She does not think she has prescription coverage so that might be cheaper.,2.  Samples of Nasonex two sprays in each nostril given for three weeks.  A prescription was written as well."
    text = "PAST MEDICAL HISTORY:, He has difficulty climbing stairs, difficulty with airline seats, tying shoes, used to public seating, and lifting objects off the floor.  He exercises three times a week at home and does cardio.  He has difficulty walking two blocks or five flights of stairs.  Difficulty with snoring.  He has muscle and joint pains including knee pain, back pain, foot and ankle pain, and swelling.  He has gastroesophageal reflux disease.,PAST SURGICAL HISTORY:, Includes reconstructive surgery on his right hand thirteen years ago.  ,SOCIAL HISTORY:, He is currently single.  He has about ten drinks a year.  He had smoked significantly up until several months ago.  He now smokes less than three cigarettes a day.,FAMILY HISTORY:, Heart disease in both grandfathers, grandmother with stroke, and a grandmother with diabetes.  Denies obesity and hypertension in other family members.,CURRENT MEDICATIONS:, None.,ALLERGIES:,  He is allergic to Penicillin.,MISCELLANEOUS/EATING HISTORY:, He has been going to support groups for seven months with Lynn Holmberg in Greenwich and he is from Eastchester, New York and he feels that we are the appropriate program.  He had a poor experience with the Greenwich program.  Eating history, he is not an emotional eater.  Does not like sweets.  He likes big portions and carbohydrates.  He likes chicken and not steak.  He currently weighs 312 pounds.  Ideal body weight would be 170 pounds.  He is 142 pounds overweight.  If ,he lost 60% of his excess body weight that would be 84 pounds and he should weigh about 228.,REVIEW OF SYSTEMS: ,Negative for head, neck, heart, lungs, GI, GU, orthopedic, and skin.  Specifically denies chest pain, heart attack, coronary artery disease, congestive heart failure, arrhythmia, atrial fibrillation, pacemaker, high cholesterol, pulmonary embolism, high blood pressure, CVA, venous insufficiency, thrombophlebitis, asthma, shortness of breath, COPD, emphysema, sleep apnea, diabetes, leg and foot swelling, osteoarthritis, rheumatoid arthritis, hiatal hernia, peptic ulcer disease, gallstones, infected gallbladder, pancreatitis, fatty liver, hepatitis, hemorrhoids, rectal bleeding, polyps, incontinence of stool, urinary stress incontinence, or cancer.  Denies cellulitis, pseudotumor cerebri, meningitis, or encephalitis.,PHYSICAL EXAMINATION:, He is alert and oriented x 3.  Cranial nerves II-XII are intact.  Afebrile.  Vital Signs are stable."
    patient_name = "Patient#1"

    patient_sections_dict, sections_triples_dict = create_triples_dict(patient_name,text)

    graph_builder = GraphBuilder()
    graph_builder.reset_graph()
    # build_graph(graph_builder, dict)
    build_new_graph(patient_sections_dict, sections_triples_dict)
    graph_builder.index_graph()

    vextor_index_name='transcription_vector_index'
    # The label of the nodes for which we want to create embeddings
    node_label='Entity'
    #Properties of the nodes containing the textual data from which embeddings will be generated
    text_node_properties=['name', 'type', 'graph']
    #The property name where the embeddings will be stored in the database
    embedding_node_property='transcription_embedding'
    embeddings_model = "text-embedding-3-small"
    # delete_vector_embeddings_and_vector_index(vextor_index_name, node_label, embedding_node_property, text_node_properties)
    transcription_vector_index = create_vector_embeddings_and_vector_index(embeddings_model,vextor_index_name, node_label, text_node_properties, embedding_node_property)  

    # graph = get_all_nodes()
    
    tools = [
        Tool(
            name="Query",
            func=query_db_by_section,
            description="Use this tool to find entities that can be used to respond to user queries"
        )
    ]

    tool_names = [f"{tool.name}: {tool.description}" for tool in tools]

    prompt_template = """
        You are a helpful agent designed to fetch information from a knowledge base containing
        information about patient transcriptions. 
    
        A typical patient transcription consists of the sections such as: 
        Past Medical History, Surgical History, Social History, Family History, Current Medications,
        Allergies, Miscellaneous/Eating History, Review of Systems, Physical Examination etc.

        Each section contains addional information about the patient's health.

        The context below returns sections relevant to the user's query. Answer the question based only on the following context:
        {context}
        
        Use natural language and be terse. Do NOT mention technical details about graphs, nodes etc. in your answer.
        Do not make any guesses or assumptions. If the information is not available in the context, 
        simply state that it is not available.
        """

    # prompt = CustomPromptTemplate(
    #     template=prompt_template,
    #     tools=tools,
    #     input_variables=["context"],
    # )
    # Initialize the transcription vector index
    # vector_index = Neo4jVector.from_existing_index(
    #     OpenAIEmbeddings(),
    #     url=os.environ["NEO4J_URI"],
    #     username=os.environ["NEO4J_USERNAME"],
    #     password=os.environ["NEO4J_PASSWORD"],
    #     index_name="transcription_vector_index",
    #     text_node_property=text_node_properties
    # )
    # # Perform similarity search on the transcription vector index
    # result = vector_index.similarity_search("past surgical history")
    # pprint(result[0].page_content)
    # #wait for user input
    # input("Press Enter to continue...")

    graph = Neo4jGraph(
        url=os.environ["NEO4J_URI"], 
        username=os.environ["NEO4J_USERNAME"], 
        password=os.environ["NEO4J_PASSWORD"])

    # # Wrap the original retriever with the custom retriever
    # custom_retriever = CustomRetriever(transcription_vector_index.as_retriever())

    prompt = PromptTemplate(input_variables=["tools", "context"], template=prompt_template)

    vector_qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=transcription_vector_index.as_retriever(), chain_type_kwargs={"prompt": prompt})

    #Cypher chain causes "too many tokens" error
    # cypher_chain = GraphCypherQAChain.from_llm(
    # cypher_llm = ChatOpenAI(temperature=0, model_name='gpt-4'),
    # qa_llm = ChatOpenAI(temperature=0), graph=graph, verbose=True,allow_dangerous_requests=True)

    questions = """
    1. Provide a summary of the health status of the patient {CANDIDATE}
    2. What is the past medical history of patient {CANDIDATE}?
    3. What is the past surgical history of patient {CANDIDATE}?
    4. What is the social history of patient {CANDIDATE}?
    5. What is the family history of patient {CANDIDATE}?
    6. What are the current medications of patient {CANDIDATE}?
    7. What are the allergies of patient {CANDIDATE}?
    8. What is the miscellaneous/eating history of patient {CANDIDATE}?
    9. What is the review of systems for patient {CANDIDATE}?
    10. What are the physical examination details for patient {CANDIDATE}?
    """

    QUESTIONS = questions.split("\n")
    #remove empty strings from the list of questions
    QUESTIONS = [q.strip() for q in QUESTIONS if q.strip()]

    candidate = "Patient#1"
    summary_of_answers = ""
    question_number = 0

    # Format the prompt with the tools and context
    # formatted_prompt = prompt_template.format(tools="\n".join(tool_names), context=result)

    for q in QUESTIONS:
        user_query = q.replace("{CANDIDATE}", candidate)

        # Retrieve the context for the query
        context_documents = transcription_vector_index.similarity_search(user_query, k=3)
        initial_context = "\n".join([doc.page_content for doc in context_documents])

        # Extract node identifiers (e.g., names) from the initial context documents
        # node_names = [doc.metadata['name'] for doc in context_documents if 'name' in doc.metadata]
        node_names = []
        for line in initial_context.split('\n'):
            match = re.search(r'name: (\w+)', line)
            if match:
                node_names.append(match.group(1))

        # Fetch related nodes and relationships for 1 or 2 more levels
        related_context = ""
        for level in range(2):  # Adjust the range for the desired number of levels
            if not node_names:
                break
            related_nodes_and_relationships = fetch_related_nodes_and_relationships(node_names)
            related_context += "\n".join([str(record) for record in related_nodes_and_relationships])
            # Extract new node names from the related nodes
            node_names = [record['m']['name'] for record in related_nodes_and_relationships if 'name' in record['m']]

        # Combine the initial context with the additional context from related nodes
        enhanced_context = initial_context + "\n" + related_context

        # response = vector_qa.invoke({"query": question, "context": system_prompt})
        tools_str = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
        response = vector_qa.invoke({"query": user_query, "context": enhanced_context})
        # response = cypher_chain.invoke({"query": user_query, "context": system_prompt})
        answer = response.get("result")
        summary_of_answers += "\nQuestion: " + user_query + "\n"
        #result_1['source_documents'] returns a list of references[0]['text']
        summary_of_answers += f"Answer: " + answer + "\n"
        # chat_history.append(HumanMessage(content=user_query))
        # chat_history.append(AIMessage(content=answer))

        question_number+=1

    print(summary_of_answers)