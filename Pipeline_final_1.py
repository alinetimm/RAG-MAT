import json
from transformers import BertTokenizer, BertModel
import torch
import pickle
from sentence_transformers import SentenceTransformer, util
import numpy as np
import logging
import faiss
import google.generativeai as genai

# ==================== CONFIGURAÇÕES ====================
# Adicione sua chave de acesso da API da Gemini aqui
API_KEY = "AIzaSyCf1dv0eXD7dnesfY0dvUUl3kIl4po5aYQ"  # Substitua pela sua chave de API real
genai.configure(api_key=API_KEY)

# Carregar o modelo e tokenizer do BERT para busca FAISS
embedding_model = BertModel.from_pretrained('bert-base-uncased')
embedding_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def generate_answer(query, context):
    """
    Gera uma resposta para uma consulta usando a Gemini.
    Args:
        query (str): A pergunta feita pelo usuário.
        context (str): O contexto relevante para a pergunta.
    Returns:
        str: Resposta gerada pela Gemini.
    """
    prompt = f"""
    Você é um assistente especializado em matemática. Usando apenas as informações fornecidas no contexto abaixo, responda à pergunta claramente e de forma amigável.
    Contexto: {context}
    Pergunta: {query}
    Resposta:
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

def evaluate_answer(query, answer):
    """
    Avalia a resposta gerada com base na pergunta original.
    Args:
        query (str): A pergunta original do usuário.
        answer (str): A resposta gerada pela IA.
    Returns:
        str: Avaliação da resposta.
    """
    prompt = f"""
    Você é uma IA avaliadora. Avalie a qualidade da resposta abaixo, considerando a pergunta feita.
    Pergunta: {query}
    Resposta: {answer}
    Avaliação:
    1. A resposta está correta? (Sim/Não)
    2. A resposta é clara e bem estruturada? (Sim/Não)
    3. Dê uma pontuação de 0 a 10 para a resposta, explicando brevemente a razão da nota.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

def search_in_faiss(query, index_path="embeddings_index.faiss", mapping_path="documents_ids.pkl", k=10):
    """
    Busca no índice FAISS os chunks mais relevantes para a consulta.
    """
    try:
        # Carregar o índice FAISS
        index = faiss.read_index(index_path)
        
        # Carregar o mapeamento de documentos
        with open(mapping_path, "rb") as f:
            mapping = pickle.load(f)
        
        # Criar embedding da query
        inputs = embedding_tokenizer(query, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = embedding_model(**inputs)
        query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        
        # Pesquisar no FAISS
        query_embedding = np.array([query_embedding], dtype="float32")
        distances, indices = index.search(query_embedding, k)
        
        # Recuperar chunks relevantes
        relevant_chunks = []
        for idx in indices[0]:
            if idx < len(mapping["documents"]):
                relevant_chunks.append(mapping["documents"][idx])
        
        # Retornar os chunks
        return relevant_chunks
    except Exception as e:
        logging.error(f"Erro ao buscar no FAISS: {e}")
        return []

def generate_response(query):
    # Recupera os chunks relevantes com base na consulta
    relevant_chunks = search_in_faiss(query)
    
    # Se não houver nenhum chunk relevante, retorna uma mensagem de erro
    if not relevant_chunks:
        return "Desculpe, não consegui encontrar informações relevantes para responder à sua pergunta."
    
    context = "\n".join(relevant_chunks[:100])
    
    response = generate_answer(query, context)

    # Avaliar a resposta
    evaluation = evaluate_answer(query, response)
    
    return response, evaluation

def process_response(response):
    cleaned_response = response.replace("\n", " ").strip()
    cleaned_response = ' '.join(cleaned_response.split())
    return cleaned_response

def chat_interface():
    print("👋 Olá! Eu sou a Galileu, e vou te ajudar com dúvidas sobre matemática!")
    print("(Digite 'sair' para encerrar ou 'contexto' para ver os documentos recuperados)")
    print("-" * 300)
    
    while True:
        try:
            query = input("\n😊 Sua pergunta: ").strip()
            if query.lower() == 'sair':
                print("\n👋 Tchau! Foi um prazer ajudar! Até a próxima!")
                break
            elif query.lower() == 'contexto':
                print("\n🌟 Contexto recuperado:", search_in_faiss(query))
                continue
            
            print("\n🤔 Pensando...")
            response, evaluation = generate_response(query)
            cleaned_response = process_response(response)
            print("\n📚 Resposta:", cleaned_response)
            print("\n📝 Avaliação da Resposta:", evaluation)
            print("-" * 50)
        except KeyboardInterrupt:
            print("\n\n👋 Tchau! Foi um prazer ajudar! Até a próxima!")
            break
        except Exception as e:
            logging.error(f"Erro no chatbot: {str(e)}")
            print("\n😅 Ops! Tive um problema ao processar sua solicitação. Por favor, tente novamente.")

if __name__ == "__main__":
    chat_interface()
