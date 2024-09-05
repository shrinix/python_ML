from flask import Flask, request, jsonify
from LLM.Chat_with_PDFs.EquityResearchReports.backend.ER_chat_service import chat, load_data

app = Flask(__name__)

# Example route
@app.route('/')
def home():
    return "Welcome to the Flask REST API!"

@app.route('/process', methods=['POST'])
def process():
    input_data = request.json.get('input_data')
    if input_data is None:
        return jsonify({"error": "No input data provided"}), 400
    
    result = chat(input_data)
    return jsonify({"result": result})

if __name__ == '__main__':
    data_dir = "../pdf-data/"
    load_data(data_dir)
    app.run(debug=True)