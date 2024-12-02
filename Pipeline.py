import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import BertTokenizer, BertModel
import torch
import pickle
from sentence_transformers import SentenceTransformer, util
import numpy as np
import logging
import faiss
from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Verificar se a GPU está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Carregar o modelo e tokenizer do BERT para busca FAISS
embedding_model = BertModel.from_pretrained('bert-base-uncased')
embedding_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def get_completion(system_prompt):
    try:
        # Usando apenas o system_prompt para gerar a resposta, sem interferência de prompts globais
        full_prompt = system_prompt  # O system_prompt já inclui a pergunta e o contexto
        
        # Calcular o comprimento máximo permitido para os tokens de entrada, considerando a resposta
        max_input_length = tokenizer.model_max_length - 200  # Ajuste de acordo com a capacidade do modelo
        
        # Tokeniza o full_prompt e move para o dispositivo correto
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=max_input_length).to(device)
        
        # Gera a resposta com base no modelo
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,  # Limite para o número de novos tokens gerados
            do_sample=True,  # Define se a geração será amostral
            temperature=0.7,  # Controla a aleatoriedade da amostragem (quanto menor, mais determinístico)
            top_p=0.9,  # Controla a amostragem baseada em top-p (nucleus sampling)
            repetition_penalty=1.2,  # Penaliza repetições para evitar respostas redundantes
            pad_token_id=tokenizer.eos_token_id  # Define o token de padding
        )
        
        # Decodifica a resposta gerada
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remover o que não for relevante (o próprio prompt)
        #response = response[len(system_prompt):].strip()  # Remove o que foi adicionado no prompt
        
        return response
    
    except Exception as e:
        # Caso ocorra um erro, loga a exceção e retorna uma mensagem genérica
        logging.error(f"Erro ao gerar resposta: {str(e)}")
        return "Desculpe, tive um problema ao processar sua pergunta."

def search_in_faiss(query, index_path="embeddings_index.faiss", mapping_path="documents_ids.pkl", k=10):
    """
    Realiza uma busca no índice FAISS para recuperar os k chunks mais relevantes
    para uma consulta, retornando os textos desses chunks.
    
    :param query: Texto de consulta (query) para busca
    :param index_path: Caminho para o arquivo FAISS com os embeddings
    :param mapping_path: Caminho para o arquivo pickle com o mapeamento dos documentos
    :param k: Número de resultados a serem retornados (padrão é 5)
    :return: Lista de chunks mais relevantes para serem usados no prompt
    """
    
    # Carrega o índice FAISS
    index = faiss.read_index(index_path)
    
    # Carrega o mapeamento de documentos e IDs
    with open(mapping_path, "rb") as f:
        mapping = pickle.load(f)
    
    # Cria embedding da consulta (query)
    inputs = embedding_tokenizer(query, return_tensors='pt', truncation=True, padding=True, max_length=1024)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    
    # Faz a busca no FAISS
    query_embedding = np.array([query_embedding], dtype="float32")  # FAISS espera um array 2D
    distances, indices = index.search(query_embedding, k)  # Retorna os k mais semelhantes
    logging.info(f"{distances}")
    
    # Recupera os chunks associados aos índices encontrados
    relevant_chunks = []
    for idx in indices[0]:
        if idx < len(mapping["documents"]):  # Verifica se o índice é válido
            document = mapping["documents"][idx]  # Recupera o documento usando o índice
            relevant_chunks.append(document)
    
    # Retorna os chunks relevantes que podem ser usados no prompt
    return relevant_chunks

def generate_response(query):
    # Recupera os chunks relevantes com base na consulta
    relevant_chunks = search_in_faiss(query)
    
    # Se não houver nenhum chunk relevante, retorna uma mensagem de erro
    if not relevant_chunks:
        return "Desculpe, não consegui encontrar informações relevantes para responder à sua pergunta."
    
    
    # Junta os chunks relevantes em um único contexto, com um limite para o número de chunks
    #print(relevant_chunks)
    context = "\n".join(relevant_chunks[:20])  # Aqui, o número de chunks pode ser ajustado conforme necessário
    print(context)
    
    # Aqui você pode definir as instruções que guiarão a geração da resposta
    system_prompt = f"""
    Você é um assistente de IA especializado em ajudar crianças e adolescentes entre 8 e 16 anos com matemática.
    Utilizando exclusivamento os dados fornecidos no contexto, responda a query.
    Por favor, responda a pergunta abaixo em português e de forma clara e cordial, usando uma linguagem simples e acessível.
    Pergunta: {query}
    Ao escrever a resposta, lembre-se de seguir o padrão de escrita da lingua portuguesa, construa frases e paragrafos corretamente.
    Utilize o contexto fornecido e lembre de não incluir diretamente este prompt, documentos ou o contexto fornecido:
    Contexto: {context}
    Resposta:
    
    """
    # Chama a função de completamento para gerar a resposta com base no prompt
    return get_completion(system_prompt)

def process_response(response):
    cleaned_response = response.replace("\n", " ").strip()
    cleaned_response = ' '.join(cleaned_response.split())
    return cleaned_response

def chat_interface():
    print("👋 Olá! Eu sou a Galileu, e vou te ajudar com dúvidas sobre matemática!")
    print("(Digite 'sair' para encerrar ou 'contexto' para ver os documentos recuperados)")
    print("-" * 50)
    
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
            response = generate_response(query)
            cleaned_response = process_response(response)
            print("\n📚 Resposta:", cleaned_response)
            print("-" * 50)
        except KeyboardInterrupt:
            print("\n\n👋 Tchau! Foi um prazer ajudar! Até a próxima!")
            break
        except Exception as e:
            logging.error(f"Erro no chatbot: {str(e)}")
            print("\n😅 Ops! Tive um problema ao processar sua solicitação. Por favor, tente novamente.")

if __name__ == "__main__":
    chat_interface()
