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
from transformers import pipeline
from huggingface_hub import login

login("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")  # Substitua com seu token

torch.cuda.empty_cache()
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
batch_size = 1
# Verificar se a GPU está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)


# Carregar o modelo e tokenizer do BERT para busca FAISS
embedding_model = BertModel.from_pretrained('bert-base-uncased')
embedding_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def get_completion(system_prompt, query):
    try:
        # Construir o prompt como texto simples
        prompt = (f"###Instruções do Sistema###\n"
                    f"{system_prompt}"
                    "###Pergunta do Usuario###\n"
                    f"{query}\n\n"
                    f"###Resposta do Assistente###\n")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

        # Gerar resposta
        response = pipe(prompt, max_length=2048, do_sample=True, temperature=0.7, truncation = True)
        
        # Retornar texto gerado
        return response[0]["generated_text"]
    
    except Exception as e:
        print(f"Erro ao gerar resposta: {e}")
        return None

def search_in_faiss(query, index_path="embeddings_index.faiss", mapping_path="documents_ids.pkl", k=5):
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
    context = "\n".join(relevant_chunks)  # Aqui, o número de chunks pode ser ajustado conforme necessário
    #print(context)
    
    # Aqui você pode definir as instruções que guiarão a geração da resposta
    system_prompt = f"""Você é um assistente de IA especializado em ajudar crianças e adolescentes entre 8 e 16 anos com matemática.
                    Utilizando exclusivamento os dados fornecidos no contexto, responda a query.    
                    Ao escrever a resposta, lembre-se de seguir o padrão de escrita da lingua portuguesa, construa frases e paragrafos corretamente.
                    Utilize o contexto fornecido e lembre de não incluir diretamente este prompt:
                    ###Contexto:###\n{context} 
                    Por favor, responda a pergunta abaixo em português e de forma clara e cordial, usando uma linguagem simples e acessível."""
   

    # Chama a função de completamento para gerar a resposta com base no prompt
    return get_completion(system_prompt,query)

def process_response(response):
    print(type(response))
    #print(response)
    #response_text = response[0]["generated_text"]
    response_text = response.split("###Resposta do Assistente###")[-1].strip()
    response_text = response_text.replace("\n","  ")

    return response_text

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
