from pprint import pprint

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
from tabulate import tabulate
from db_schema import get_db_schema
from pglast.parser import parse_sql, ParseError

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
    pprint(prompt)

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

def is_valid_query(query: str) -> bool:
    """Validates query syntax using Postgres parser.

    Note: in this context, "invalid" includes a query that is empty or only a
    SQL comment, which is different from the typical sense of "valid Postgres".
    """
    parse_result = None
    valid = True
    try:
        parse_result = parse_sql(query)
    except ParseError as e:
        valid = False
    # Check for any empty result (occurs if completion is empty or a comment)
    return parse_result and valid

import re

import re

def extract_select_statement(text):
    pattern = r"(SELECT.*?);"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return None

def main():

    # Defining the models to be used
    LLM = ChatOllama(model='sqlcoder')
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

        #Query validation using pglast
        query = extract_select_statement(response)
        print(query)
        valid = is_valid_query(query)

        if(valid):
            print("Valid query")
            # execute_query(query, cur)
        else:
            print("Invalid query")

        # import psycopg2
        # conn = psycopg2.connect(
        #     dbname=mydatabase,
        #     user=username,
        #     password=password,
        #     host="localhost",
        #     port="5432"
        # )

if __name__ == "__main__":
    main()

