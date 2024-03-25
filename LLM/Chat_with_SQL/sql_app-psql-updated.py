from langchain_community.llms import CTransformers
from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain_community.chat_models import ChatOllama
import re
from db_connectors import PostgresConnector
import psycopg2
from tabulate import tabulate

def generate_prompt(question, prompt_file="prompt.md", metadata_file="metadata.sql"):
    with open(prompt_file, "r") as f:
        prompt = f.read()

    with open(metadata_file, "r") as f:
        table_metadata_string = f.read()

    prompt = prompt.format(
        user_question=question, table_metadata_string=table_metadata_string
    )

    prompt = PromptTemplate.from_template(
        prompt, metadata={"user_question": question, "table_metadata_string": table_metadata_string}
    )
    return prompt

def query_model(question, llm):
    prompt = generate_prompt(question)

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    # output = llm(
    #     prompt,
    #     max_tokens=200, # Generate up to 32 tokens
    #     stop=["Q:", "\n"],
    #     echo=True # Echo the prompt back in the output
    # )

    response = chain.invoke(
        {"input":prompt}
    )

   # sql_query = response['choices'][0]['text'].split('[SQL]')[1]
    print("SQL: ", response)

    return response
    #return execute_query(response)

def execute_query(query, cur):
    cur.execute(query)
    rows = cur.fetchall()
    headers = [desc[0] for desc in cur.description]
    return tabulate(rows, headers=headers, tablefmt="grid")

def main():

    # llm = CTransformers(model="//Users//shriniwasiyengar//.cache//lm-studio//models//TheBloke//nsql-llama-2-7B-GGUF",
    #                     model_type="llama",
    #                     config={'max_new_tokens': 512,
    #                             'temperature': 0.01,
    #                             'context_length': 6000})

    # Defining the models to be used
    LLM = ChatOllama(model='sqlcoder')
    #llm = ChatOllama(model='sqlcoder', config={'max_new_tokens': 512, 'temperature': 0.01, 'context_length': 6000})

    from huggingface_hub import snapshot_download
    # model_path = snapshot_download(repo_id="defog/sqlcoder-7b-2", repo_type="model",
    #                                local_dir="../models/sqlcoder-7b-2", local_dir_use_symlinks=False)
    host = 'localhost'
    port = '5432'
    username = 'postgres'
    password = 'New Password'
    mydatabase = 'dvdrental'

    pg_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{mydatabase}"

    db = SQLDatabase.from_uri(pg_uri,
                              include_tables=['actor', 'address', 'category', 'city', 'country', 'customer', 'film',
                                              'film_actor', 'film_category', 'inventory', 'language', 'payment',
                                              'rental', 'staff', 'store'],
                              sample_rows_in_table_info=2)
    table_info = db.get_table_info()
    #print(db.table_info)
    dialect = db.dialect
    print("Welcome to the SQLCoder Chat Interface!")
    print("Type your message and press Enter to get a response.")
    print("Type 'exit' to quit.")

    # conn = psycopg2.connect(
    #     dbname="yelp",
    #     user="postgres",
    #     password="postgres",
    #     host="localhost",
    #     port="5432"
    # )
    # cur = conn.cursor()
    #

    # from llama_cpp import Llama
    import psycopg2
    from tabulate import tabulate

    #LLM = Llama(model_path="/Users/shriniwasiyengar/Downloads/sqlcoder-7b-q5_k_m.gguf", n_gpu_layers=10, n_ctx=2048,
     #           verbose=False)

    while True:
        user_input = input("Question: ")
        if user_input.lower() == 'exit':
            print("Exiting...")
            # cur.close()
            # conn.close()
            break
        print("Answer: ")
        response = query_model(user_input, LLM)
        print(response)

if __name__ == "__main__":
    main()

