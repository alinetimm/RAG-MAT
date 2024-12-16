from flask import Flask, request, render_template, jsonify
from flask_cors import CORS  
import logging
import os
from dotenv import load_dotenv
from Pipeline_galileu import generate_response, process_response  

app = Flask(__name__)
CORS(app)
API_KEY = "AIzaSyCf1dv0eXD7dnesfY0dvUUl3kIl4po5aYQ"

try:
    load_dotenv(encoding='utf-16')
except UnicodeDecodeError as e:
    logging.error(f"Erro de codificação ao carregar o arquivo .env: {str(e)}")
except Exception as e:
    logging.error(f"Erro ao carregar o arquivo .env: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    # Verifica se a requisição é JSON ou form-data
    if request.is_json:
        data = request.get_json()
        query = data.get('query', '')
    else:
        query = request.form.get('query', '')

    # Verifica se a query está vazia
    if not query.strip():
        return jsonify({"response": "Por favor, insira uma pergunta.", "evaluation": ""})
    
    try:
        # Processa a resposta
        response, evaluation = generate_response(query)
        cleaned_response = process_response(response)
        result = {
            "response": cleaned_response,
            "evaluation": evaluation
        }
    except Exception as e:
        logging.error(f"Erro ao processar a consulta: {str(e)}")
        result = {
            "response": "😅 Ops! Tive um problema ao processar sua solicitação. Por favor, tente novamente.",
            "evaluation": ""
        }
    
    return jsonify(result)

@app.route('/chat', methods=['POST'])
def chat_route():  # Renomeei de 'chat' para 'chat_route'
    data = request.get_json()  # Obtém os dados JSON da requisição
    query = data.get('query', '')  # Obtém a consulta do JSON
    
    if not query.strip():
        return jsonify({"response": "Por favor, insira uma pergunta.", "evaluation": ""})
    
    try:
        response, evaluation = generate_response(query)
        cleaned_response = process_response(response)
        result = {
            "response": cleaned_response,
            "evaluation": evaluation
        }
    except Exception as e:
        logging.error(f"Erro ao processar a consulta: {str(e)}")
        result = {
            "response": "😅 Ops! Tive um problema ao processar sua solicitação. Por favor, tente novamente.",
            "evaluation": ""
        }
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
