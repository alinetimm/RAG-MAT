import requests
import pdfplumber
import pickle
from sentence_transformers import SentenceTransformer
import io
import nltk
from PyPDF2 import PdfReader

# Baixar recursos do NLTK
nltk.download('punkt')

def get_direct_download_url(gdrive_url):
    """Extrai o ID do arquivo do URL do Google Drive."""
    file_id = gdrive_url.split('/d/')[1].split('/view')[0]
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def extract_text_from_pdf_url(url):
    """Lê e processa o PDF diretamente de uma URL."""
    try:
        direct_url = get_direct_download_url(url)
        response = requests.get(direct_url)
        response.raise_for_status()
        
        pdf_stream = io.BytesIO(response.content)
        text = ""
        with pdfplumber.open(pdf_stream) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                # Remover quebras de linha desnecessárias
                page_text = page_text.replace('\n', ' ')
                text += page_text + " "
        
        return text
    except Exception as e:
        print(f"Erro ao baixar ou ler o PDF: {e}")
        return ""

def process_pdf(file_path):
    """Lê e processa o PDF, retornando o texto por página como chunks."""
    reader = PdfReader(file_path)
    chunks = []

    # Itera sobre cada página do PDF
    for page in reader.pages:
        text = page.extract_text()
        if text:
            # Divide o texto em parágrafos ou sentenças
            paragraphs = text.split('\n\n')  # Dividir por parágrafos
            for paragraph in paragraphs:
                if paragraph.strip():
                    chunks.append(paragraph.strip())  # Adiciona o parágrafo como um chunk

    return chunks

def process_multiple_pdfs_and_create_embeddings(pdf_urls):
    print("Iniciando processamento dos PDFs e criação de embeddings...")
    
    # Carrega o modelo de embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    all_documents = []
    all_embeddings = []
    
    for pdf_url in pdf_urls:
        # Extrai texto do PDF
        texto_completo = extract_text_from_pdf_url(pdf_url)
        
        if not texto_completo:
            print(f"Erro: Nenhum texto foi extraído do PDF {pdf_url}!")
            continue
        
        # Processa o texto em chunks
        documents = process_pdf(texto_completo)  
        
        print(f"Extraídas {len(documents)} chunks do arquivo {pdf_url}")
        
        # Cria embeddings
        print("\nCriando embeddings...")
        embeddings = model.encode(documents, show_progress_bar=True)
        
        # Armazena os documentos e embeddings
        all_documents.extend(documents)
        all_embeddings.extend(embeddings)
    

    # Salva os resultados
    with open("embeddings_data.pkl", "wb") as f:
        pickle.dump({
            "documents": all_documents,
            "embeddings": all_embeddings
        }, f)
    
    print("Processamento concluído!")

    if len(all_documents) != len(all_embeddings):
        print("Erro: O número de documentos e embeddings não coincide!")

# Lista de URLs dos PDFs que você deseja processar
pdf_urls = [
    "https://drive.google.com/file/d/13UhN1HXF5HUby5RSC4eYZaWHik0olsZc/view?usp=drive_link",
    "https://drive.google.com/file/d/1o5X22MuBYxnGsTkdQ602DGCBYrO-M1kR/view?usp=sharing",
    "https://drive.google.com/file/d/1UUTw1b3e79Y6iO457RoMYole9YBs7pVh/view?usp=sharing",
    "https://drive.google.com/file/d/1UKepm6IWBdUDpOtuuXFGaI98sOaqH0uP/view?usp=sharing",
    "https://drive.google.com/file/d/18IpxyImhLiMtM-ZKFnD7T3biT-QqtWnI/view?usp=drive_link"
]

if __name__ == "__main__":
    process_multiple_pdfs_and_create_embeddings(pdf_urls)

# Ao carregar os embeddings
with open("embeddings_data.pkl", "rb") as f:
    data = pickle.load(f)
    all_documents = data["documents"]
    all_embeddings = data["embeddings"]

print(f"Número de documentos carregados: {len(all_documents)}")
print(f"Número de embeddings carregados: {len(all_embeddings)}")

if len(all_documents) != len(all_embeddings):
    print("Erro: O número de documentos e embeddings não coincide!")
