"""
Module to manage all GraphRAG functions. Mainly revolves around building up 
a retriever that can parse knowledge graphs and return related results
"""
from typing import List
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from entities import Entities
import re
from FlagEmbedding import FlagReranker
import numpy as np
import logging as logger

class GraphRAG():
    """
    Class to encapsulate all methods required to query a graph for retrieval augmented generation
    """

    def __init__(self):
        self.graph = Neo4jGraph()
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def create_entity_extract_chain(self):
        """
        Creates a chain which will extract entities from the question posed by the user. 
        This allows us to search the graph for nodes which correspond to entities more efficiently

        Returns:
            Runnable: Runnable chain which uses the LLM to extract entities from the users question
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are extracting objects, person, medical information, organization, " +
                    "or business entities from the text.",
                ),
                (
                    "human",
                    "Use the given format to extract information from the following "
                    "input: {question}",
                ),
            ]
        )

        entity_extract_chain = prompt | self.llm.with_structured_output(Entities)
        return entity_extract_chain

    def generate_full_text_query(self, input_query: str) -> str:
        """
        Generate a full-text search query for a given input string.

        This function constructs a query string suitable for a full-text search.
        It processes the input string by splitting it into words and appending a
        similarity threshold (~2 changed characters) to each word, then combines
        them using the AND operator. Useful for mapping entities from user questions
        to database values, and allows for some misspellings.

        Args:
            input_query (str): The extracted entity name pulled from the users question

        Returns:
            str: _description_
        """
        full_text_query = ""

        # split out words and remove any special characters reserved for cipher query
        words = [el for el in remove_lucene_chars(input_query).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()

    def structured_retriever(self, question: str) -> str:
        """
        Creates a retriever which will use entities extracted from the users query to 
        request context from the Graph and return the neighboring nodes and edges related
        to that query. 
        
        Args:
            question (str): The question posed by the user for this graph RAG

        Returns:
            str: The fully formed Graph Query which will retrieve the 
                 context relevant to the users question
        """

        entity_extract_chain = self.create_entity_extract_chain()
        result = ""
        entities = entity_extract_chain.invoke({"question": question})
        for entity in entities.names:
            response = self.graph.query(
                """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                YIELD node, score
                CALL {
                    WITH node
                    MATCH (node)-[r]->(neighbor)
                    RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                    UNION ALL
                    WITH node
                    MATCH (node)<-[r]-(neighbor)
                    RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
                } 
                RETURN output LIMIT 100
                """,
                {"query": self.generate_full_text_query(entity)},
            )
            result += "\n".join([el['output'] for el in response])
        return result

    def create_vector_index(self) -> Neo4jVector:
        """
        Uses the existing graph to create a vector index. This vector representation
        is based off the properties specified. Using OpenAIEmbeddings since we are using 
        GPT-4o as the model. 

        Returns:
            Neo4jVector: The vector representation of the graph nodes specified in the configuration
        """
        vector_index = Neo4jVector.from_existing_graph(
            OpenAIEmbeddings(),
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )
        return vector_index

    def retrieve_reranked_docs(self, all_result_docs, query, doc_number,  reranker):

        try:
            # results = []
            # for collection in collections:
            #     result = []
            #     print("Querying collection for reranking: ", collection.name)
            #     # Retrieve relevant results from the vector database
            #     result = collection.query(query_texts=[query], n_results=doc_number)
            #     #accumulate the results from all collections
            #    # results.extend(result["documents"])
            #     print("2")  

                # Ensure query is a string
            if not isinstance(query, str):
                raise ValueError("Query must be a string")
            all_result_docs_strings = []
            # convert the list of lists to a list of strings
            for i in range(len(all_result_docs)):
                doc_str = all_result_docs[i]
                all_result_docs_strings.append(doc_str)
            
            # Ensure all elements in all_result_docs are strings
            for doc in all_result_docs_strings:
                #convert from list to string
                if not isinstance(doc, str):
                    print("Error in all_result_docs")
                    print(doc)
                    print(type(doc))
                    raise ValueError("All elements in all_result_docs must be strings")

            # Compute relevance scores for the retrieved documents
            pairs = [[query, result] for result in all_result_docs_strings]
            scores = reranker.compute_score(pairs, normalize=True)

            # Sort the documents based on the relevance scores - higher the score, the greater the relevance
            # https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/reranker/README.md
            sorted_results = sorted(zip(all_result_docs_strings, scores), key=lambda x: x[1], reverse=True)

            # Extract the retrieved document texts and their relevance scores
            retrieved_documents = [(result, score) for result, score in sorted_results[:doc_number]]

            return retrieved_documents

        except Exception as e:
            logger.error(f"Error in retrieve_reranked_docs: {e}")
            return []

    def sanitize_query(self, query: str) -> str:
        """
        Sanitize the query string to remove or escape problematic characters for Neo4j full-text search.
        
        Args:
            query (str): The original query string.
        
        Returns:
            str: The sanitized query string.
        """
        # Remove problematic characters
        sanitized_query = re.sub(r'[^\w\s]', '', query)
        return sanitized_query

    def retriever(self, question: str) -> str:
        """
        The graph RAG retriever which combines both structured and unstructured methods of retrieval 
        into a single retriever based off the users question. 
        
        Args:
            question (str): The question posed by the user for this graph RAG

        Returns:
            str: The retrieved data from the graph in both forms
        """
        print(f"Search query: {question}")
        sanitized_question = self.sanitize_query(question)
        vector_index = self.create_vector_index()
        unstructured_data = [el.page_content for el in vector_index.similarity_search(sanitized_question)]

        ### Reranking
        # Loading the Reranking Model (Also open-sourced from Huggingface) 
        # reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
        # reranked_docs = self.retrieve_reranked_docs(unstructured_data, sanitized_question, len(unstructured_data), reranker)
        # relevance_scores = []
        # reranked_documents = []
        # # Access individual document texts and their relevance scores
        # for i, (doc_reranked_text, score) in enumerate(reranked_docs):
        #     relevance_scores.append(score)
        #     reranked_documents.append(doc_reranked_text)
        #     print(f"Document {i+1} (Relevance Score: {score:.4f}):")
        #     print(doc_reranked_text)
        #     print()

        # relevance_scores = np.round(relevance_scores,2)
        # i=0
        # for i in range(len(reranked_documents)):
        #     reranked_documents[i]+=f"\nScore: {relevance_scores[i]}"
        #     i+=1
        ### End Reranking

        structured_data = self.structured_retriever(sanitized_question)
        
        # Extract string content from tuples in reranked_docs
        # reranked_docs_strings = [doc[0] if isinstance(doc, tuple) else doc for doc in reranked_docs]

        final_data = f"""Structured data:
            {structured_data} 
            Unstructured data:
            {"#Document ". join(unstructured_data)}
        """
        return final_data

    def create_search_query(self, chat_history: List, question: str) -> str:
        """
        Combines chat history along with the current question into a prompt that 
        can be executed by the LLM to answer the new question with history.

        Args:
            chat_history (List): List of messages captured during this conversation
            question (str): The question posed by the user for this graph RAG

        Returns:
            str: The formatted prompt that can be sent to the LLM with question & chat history
        """
        search_query = ChatPromptTemplate.from_messages([
            (
                "system",
                """Given the following conversation and a follow up question, rephrase the follow 
                up question to be a standalone question, in its original language.
                Chat History:
                {chat_history}
                Follow Up Input: {question}
                Standalone question:"""
            )
        ])
        formatted_query = search_query.format(
            chat_history=chat_history, question=question)
        return formatted_query
