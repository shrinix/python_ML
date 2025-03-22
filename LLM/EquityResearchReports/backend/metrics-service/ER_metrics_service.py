from datasets import Dataset
from ragas import evaluate
from flask import Flask, request, jsonify
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall, #requires reference data to be created
    context_precision, #requires reference data to be created
    answer_correctness
)
app = Flask(__name__)

@app.route('/generate_metrics', methods=['GET', 'POST'])
def generate_metrics_endpoint():
    data = request.json
    queries = data.get('queries')
    candidates = data.get('candidates')
    contexts = data.get('contexts')
    
    if not queries or not candidates or not contexts:
        return jsonify({"error": "Invalid input"}), 400
    
    metrics = generate_metrics(queries, candidates, contexts)
    return jsonify(metrics.to_dict(orient='records'))

def generate_metrics(queries, candidates, contexts):
 
    # To dict
    data = {
        "question": queries,
        "response": candidates,
        "retrieved_contexts": contexts,
        # "ground_truths": [[]] * len(candidates) #reference data
    }
    
    # Convert dict to dataset
    dataset = Dataset.from_dict(data)

    result = evaluate(
        dataset = dataset, 
        metrics=[
            faithfulness,
            answer_relevancy,
            # answer_correctness, #requires reference data to be created
            # context_precision, #requires reference data to be created
            # context_recall, #requires reference data to be created
        ],
    )

    metrics = result.to_pandas()
    #calculate arithmetic mean of the metrics and add a new column to the dataframe
    metrics['RAGAS Score'] = (metrics['faithfulness'] + metrics['answer_relevancy'])/2
    #trim the number of decimal places to 2 for numerical columns in the dataframe
    metrics = metrics.round(2) 
    return metrics

if __name__ == '__main__':
    app.run(port=5002, debug=True, host='0.0.0.0')
