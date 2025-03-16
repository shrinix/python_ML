from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from flashrank import Ranker
from langchain_openai import ChatOpenAI
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/reranking_answer', methods=['POST'])
def reranking_answer():
    data = request.json
    question = data.get('question')
    chat_history = data.get('chat_history', [])
    vectordb = data.get('vectordb')  # Assuming vectordb is passed in the request

    if not question or not vectordb:
        return jsonify({"error": "Question and vectordb are required"}), 400

    try:
        result = answer_with_reranking(question, chat_history, vectordb)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
def answer_with_reranking(question, chat_history, vectordb):

    retriever = vectordb.as_retriever(search_kwargs={"k":10})
    ranker = Ranker(model_name='ms-marco-MultiBERT-L-12', cache_dir='.')
    compressor = FlashrankRerank(client=ranker, top_n=5)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,
                                                        base_retriever=retriever
                                                        )
    compressed_docs = compression_retriever.invoke(question)
    from langchain.chains import RetrievalQA

    llm = ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo')
    qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=compression_retriever,
                                 return_source_documents=True
                                 )
    result = qa.invoke({"query": question, "chat_history": chat_history})
    print(result)
    return result

if __name__ == '__main__':
    app.run(debug=True)
