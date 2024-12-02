import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import pickle
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Carregue o modelo e tokenizer uma vez
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Carregar embeddings e documentos
with open("embeddings_data.pkl", "rb") as f:
    data = pickle.load(f)
    all_documents = data["documents"]
    all_embeddings = data["embeddings"]

# Carregar o modelo de embeddings
embedding_model = SentenceTransformer('bert-base-uncased')

def get_completion(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    outputs = model.generate(
        **inputs,
        max_length=1200,
        min_length=100,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
        repetition_penalty=1.2
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def retrieve_documents(query, top_k=5):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, all_embeddings)[0]
    similarities = torch.tensor(np.array(similarities))

    # Garantir que top_k não seja maior que o número de documentos
    top_k = min(top_k, len(all_documents))
    top_results = torch.topk(similarities, k=top_k)
    return "\n".join(all_documents[idx] for idx in top_results.indices)

def generate_response(query):
    # Recupera documentos relevantes
    relevant_docs = retrieve_documents(query)
    
    # Cria um contexto a partir dos documentos recuperados
    context = "\n".join(relevant_docs)
    
    # Geração do prompt
    prompt = f"""
    Você é um assistente de IA especializado em ajudar crianças e adolescentes entre 8 e 16 anos com matemática.
    Utilizando exclusivamento os dados fornecidos no contexto, responda a query.
    Por favor, responda em português e de forma clara e cordial, usando uma linguagem simples e acessível.
    Ao escrever a resposta, lembre-se de seguir o padrão de escrita da lingua portuguesa, construa frases e paragrafos corretamente.
    Baseie-se no seguinte contexto para responder (não mencione o contexto na resposta):
    {context}

    Pergunta: {query}

    Responda de forma detalhada simples e objetiva.
    """

    # Chamada ao modelo para gerar a resposta
    response = get_completion(prompt)
    
    return response

def process_response(response):
    # Remover partes desnecessárias da resposta
    cleaned_response = response.split("Pergunta:")[0]  
    cleaned_response = cleaned_response.strip()  
    cleaned_response = cleaned_response.replace("\n", " ")  
    cleaned_response = ' '.join(cleaned_response.split())  
    
    # Adiciona uma etapa para garantir que o texto seja formatado corretamente (n funcionou)
    cleaned_response = cleaned_response.replace(" .", ".")  
    cleaned_response = cleaned_response.replace(" ,", ",") 
    cleaned_response = cleaned_response.replace(" ;", ";") 
    cleaned_response = cleaned_response.replace(" :", ":")  

    return cleaned_response

def chat_interface():
    print("👋 Olá! Eu sou a Galileu, e vou te ajudar com dúvidas sobre matemática!")
    print("(digite 'sair' para encerrar)")
    print("-" * 50)
    
    while True:
        try:
            query = input("\n😊 Sua pergunta: ")
            if query.lower() == 'sair':
                print("\n👋 Tchau! Foi um prazer ajudar! Até a próxima!")
                break
                
            print("\n🤔 Pensando...")
            response = generate_response(query)  # Passando a consulta
            print("\n📚 Resposta:", response)
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\n👋 Tchau! Foi um prazer ajudar! Até a próxima!")
            break
        except Exception as e:
            print(f"\n😅 Ops! Tive um problema: {str(e)}")

if __name__ == "__main__":
    chat_interface()