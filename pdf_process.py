import requests
import pdfplumber
import pickle
from transformers import BertTokenizer, BertModel
import io
import nltk
from PyPDF2 import PdfReader
import gc
import logging
import torch
import faiss
import numpy as np
import re

# Baixar recursos do NLTK
nltk.download('punkt')

def get_direct_download_url(gdrive_url):
    """Extrai o ID do arquivo do URL do Google Drive."""
    file_id = gdrive_url.split('/d/')[1].split('/view')[0]
    return f"https://drive.google.com/uc?export=download&id={file_id}"


# Configuração do log
logging.basicConfig(
    filename='pdf_processing.log',  # Nome do arquivo de log
    level=logging.INFO,  # Nível de log (INFO para logs gerais, DEBUG para mais detalhes)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Formato do log
    datefmt='%Y-%m-%d %H:%M:%S'  # Formato da data/hora
)

def extract_text_and_images_from_pdf_url(url, output_filename="output.txt"):
    """Lê e processa o PDF diretamente de uma URL, mantendo a formatação original e salva em um arquivo .txt."""
    try:
        logging.info(f"Obtendo a URL direta do PDF: {url}")
        direct_url = get_direct_download_url(url)
        logging.info(f"Baixando PDF da URL: {direct_url}")
        response = requests.get(direct_url)
        response.raise_for_status()
        
        pdf_stream = io.BytesIO(response.content)
        pdf_size_mb = len(response.content) / 1024 / 1024
        logging.info(f"Tamanho do PDF: {pdf_size_mb:.2f} MB")
        
        text = ""
        with pdfplumber.open(pdf_stream) as pdf:
            total_pages = len(pdf.pages)
            logging.info(f"PDF aberto com sucesso. Total de páginas: {total_pages}")
            
            for page in pdf.pages:
                try:
                    logging.info(f"Processando página {page.page_number}...")
                    page_text = page.extract_text() or ""
                    # Adiciona um espaço entre palavras que estão juntas
                    # page_text = re.sub(r'(?<!\s)(?=\S)', ' ', page_text)  # Adiciona espaço antes de palavras que não têm espaço
                    text += ' '.join(page_text.split()) + " "  # Substitui múltiplos espaços por um único espaço
                    
                    # Extração de imagens
                    images = page.images
                    for img in images:
                        # Salvar ou processar a imagem
                        # Exemplo: img_data = pdf.pages[page.page_number - 1].to_image()
                        # img_data.save(f"image_page_{page.page_number}.png")
                        pass  # Adicione seu código para salvar/processar imagens aqui
                    
                except Exception as page_error:
                    logging.error(f"Erro ao processar a página {page.page_number}: {page_error}")
                finally:
                    del page
                    gc.collect()
        
        logging.info("Processamento do PDF concluído com sucesso.")
        
        # Salvar o texto extraído em um arquivo .txt
        with open(output_filename, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text)
        
        logging.info(f"Texto extraído e salvo com sucesso no arquivo: {output_filename}")
        
        return text
    except Exception as e:
        logging.error(f"Erro ao baixar ou processar o PDF: {e}")
        return ""


def process_pdf(text):
    """Processa o texto de um PDF, retornando o conteúdo dividido em chunks de 300 palavras."""
    logging.info("Iniciando processamento do texto.")
    chunks = []

    try:
        # Verifica se o texto é válido
        if not text.strip():
            logging.warning("Texto vazio fornecido. Nada para processar.")
            return chunks

        # Divide o texto em sentenças
        paragraphs = text.split('.')  # Divisão inicial por sentenças
        logging.info(f"Número inicial de sentenças extraídas: {len(paragraphs)}")

        current_chunk = []
        current_word_count = 0

        # Processa cada sentença, removendo espaços e adicionando como chunks
        for paragraph in paragraphs:
            words = paragraph.strip().split()
            word_count = len(words)

            # Verifica se adicionar a sentença excede o limite de 300 palavras
            if current_word_count + word_count > 300:
                # Adiciona o chunk atual à lista de chunks
                chunks.append(' '.join(current_chunk))
                # Reinicia o chunk atual
                current_chunk = words
                current_word_count = word_count
            else:
                current_chunk.extend(words)
                current_word_count += word_count
        
        # Adiciona o último chunk se não estiver vazio
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        # Imprime os chunks extraídos
        print("Chunks extraídos:", chunks)  # Adicionado para imprimir os chunks
        
        # Verificação e log do número de chunks
        logging.info(f"Total de chunks criados: {len(chunks)}")
        
        return chunks

    except Exception as e:
        logging.error(f"Erro ao processar o texto: {e}")
        return []


import faiss
import numpy as np

import faiss
import numpy as np

def process_multiple_pdfs_and_store_in_faiss(pdf_urls):
    print("Iniciando processamento dos PDFs e criação de embeddings...")
    
    # Carrega o modelo BERT e o tokenizer
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Lista para manter os documentos e seus IDs
    all_documents = []
    document_ids = []
    embeddings_list = []
    
    for pdf_url in pdf_urls:
        # Extrai texto do PDF
        texto_completo = extract_text_and_images_from_pdf_url(pdf_url)
        
        if not texto_completo:
            print(f"Erro: Nenhum texto foi extraído do PDF {pdf_url}!")
            continue
        
        # Processa o texto em chunks
        documents = process_pdf(texto_completo)  
        print(f"Extraídas {len(documents)} chunks do arquivo {pdf_url}")
        
        # Cria embeddings
        print("\nCriando embeddings...")
        for i, document in enumerate(documents):
            # Tokeniza o documento
            inputs = tokenizer(document, return_tensors='pt', truncation=True, padding=True, max_length=512)
            with torch.no_grad():  # Desabilitar o cálculo de gradientes para otimização
                outputs = model(**inputs)
            
            # Obtém os embeddings da camada final (última camada oculta)
            last_hidden_states = outputs.last_hidden_state
            embedding = last_hidden_states.mean(dim=1).squeeze().cpu().numpy()
            
            # Adiciona o embedding e documento
            embeddings_list.append(embedding)
            all_documents.append(document)
            document_ids.append(f"{pdf_url}_chunk_{i}")
    
    # Configura o índice FAISS
    embedding_dim = embeddings_list[0].shape[0]  # Dimensão do vetor
    index = faiss.IndexFlatL2(embedding_dim)  # Index para similaridade usando L2 (distância euclidiana)
    
    # Converte a lista de embeddings para numpy array
    embeddings_array = np.array(embeddings_list, dtype="float32")
    
    # Adiciona os embeddings ao índice
    index.add(embeddings_array)
    print(f"Total de embeddings armazenados no FAISS: {index.ntotal}")
    
    # Salva o índice FAISS e os documentos associados
    faiss.write_index(index, "embeddings_index.faiss")
    with open("documents_ids.pkl", "wb") as f:
        pickle.dump({
            "document_ids": document_ids,
            "documents": all_documents,
            "embedding_model": model,
            "tokenizer": tokenizer
        }, f)
    
    print("Processamento e armazenamento no FAISS concluídos!")



pdf_urls = [
    "https://drive.google.com/file/d/13UhN1HXF5HUby5RSC4eYZaWHik0olsZc/view?usp=drive_link",
    "https://drive.google.com/file/d/1nG_7_IrlIH-tdQKy5Lmg47ced-ntPwu9/view?usp=sharing",
    "https://drive.google.com/file/d/1vA4CfpR1QmEjhcXtL4Mwbhj5QbcySth5/view?usp=sharing",
    #"https://drive.google.com/file/d/1UUTw1b3e79Y6iO457RoMYole9YBs7pVh/view?usp=sharing",
    #"https://drive.google.com/file/d/1UKepm6IWBdUDpOtuuXFGaI98sOaqH0uP/view?usp=sharing",
]

if __name__ == "__main__":
    process_multiple_pdfs_and_store_in_faiss(pdf_urls)

import faiss
print(faiss.__version__)

