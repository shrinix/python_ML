from langchain_community.llms import CTransformers
from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain_community.chat_models import ChatOllama
import re
from db_connectors import PostgresConnector

if __name__ == '__main__':


    # llm = CTransformers(model="//Users//shriniwasiyengar//.cache//lm-studio//models//TheBloke//nsql-llama-2-7B-GGUF",
    #                     model_type="llama",
    #                     config={'max_new_tokens': 512,
    #                             'temperature': 0.01,
    #                             'context_length': 6000})



    # Defining the models to be used
    #llm = ChatOllama(model='sqlcoder')
    llm = ChatOllama(model='llama2', config={'max_new_tokens': 512, 'temperature': 0.01, 'context_length': 6000})

    host = 'localhost'
    port = '5432'
    username = 'postgres'
    password = 'New Password'
    mydatabase = 'dvdrental'

    pg_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{mydatabase}"

    db = SQLDatabase.from_uri(pg_uri,
                              include_tables=['actor', 'address', 'category', 'city', 'country', 'customer', 'film', 'film_actor', 'film_category', 'inventory', 'language', 'payment', 'rental', 'staff', 'store'],
                              sample_rows_in_table_info=2)
    table_info = db.get_table_info()
    #print(db.table_info)
    dialect = db.dialect

    #write a loop to get all the columns in each of the tables in table_info
    # for table in table_info:
    #     print(PostgresConnector(pg_uri).get_keys(table))

    system = (
        f"You are a {dialect} expert. Given an input question, create a syntactically correct {dialect} query to run."
        "Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per {dialect}. You can order the results to return the most informative data in the database."
        "Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (\") to denote them as delimited identifiers."
        "Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table."
        "Pay attention to use date('now') function to get the current date, if the question involves \"today\"."

        "Only use the following tables and columns based on the schema below:"
        f"{table_info}"

        f"Write an initial draft of the query. Then double check the {dialect} query for common mistakes, including:"
        "- Using NOT IN with NULL values"
        "- Using UNION when UNION ALL should have been used"
        "- Using BETWEEN for exclusive ranges"
        "- Data type mismatch in predicates"
        "- NOT Properly quoting identifiers"
        "- NOT Using the correct number of arguments for functions"
        "- NOT Casting to the correct data type"
        "- NOT Using the proper columns for joins"
        "- NOT Using the correct table aliases"
        "- NOT Using the correct table names as defined in the schema"
        "- NOT Using the correct column names as defined in the schema"
        "- NOT Using correct column names in GROUP BY and ORDER BY clauses"
        "- NOT Keeping the queries simple and easy to understand"
        "- NOT Using the correct syntax for the {dialect} query"
        "- Using unnecessary subqueries and conditions"

        "Use format:"
        "Final answer: <<FINAL_ANSWER_QUERY>>")

    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "{input}")])

    #prompt = prompt.input_variables(dialect=db.dialect, table_info=table_info)
    prompt.input_variables.append('table_info')
    # print(prompt.input_variables[0])
    # print(type(prompt.input_variables))

    question = "How many films have film length was more than 60 minutes?"
    #question = "Find the total number of actors"
    #question = "Movies in how many different languages are available in the store?"
    #question = "What is the total number of films in the store?"

    # input = question

    def parse_final_answer(text):
        print(text)
        pattern = r"\n```(.*?)\n```\n"
        match = re.search(pattern, text, re.DOTALL)

        if match:
            result = match.group(1)
            print(result)
        return result

    chain = create_sql_query_chain(llm, db, prompt=prompt)
    #prompt.pretty_print()

    answer = chain.invoke(
        {
            "question": question
        }
    )
    print(answer)
    print("-----------")
    print("-----------")
    print("-----------")
    system2 = (
        f"Cross-reference the {answer} against the schema below."
        f"{table_info}"

        f"Double check the query for the following:"
        f" -Correct SQL syntax as per {dialect}"
         "- Ensure that the table names in the query are in the list ['actor', 'address', 'category', 'city', 'country', 'customer', 'film', 'film_actor', 'film_category', 'inventory', 'language', 'payment', 'rental', 'staff', 'store']"         
        
        "Only return the final corrected SQL query")

    prompt2 = ChatPromptTemplate.from_messages(
        [("system", system2), ("human", "{input}")])

    # prompt = prompt.input_variables(dialect=db.dialect, table_info=table_info)
    prompt2.input_variables.append('table_info')
    prompt2.input_variables.append('top_k')
    query1 = parse_final_answer(answer)

    chain2 = create_sql_query_chain(llm, db, prompt=prompt2)
    query2 = chain2.invoke(
        {
            "question": query1
        }
    )
    print(query2)

