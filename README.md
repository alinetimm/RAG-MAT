# RAG para Respostas Matemáticas Usando a LLM Gemini

Este projeto implementa um sistema **RAG (Retrieval-Augmented Generation)** para responder a perguntas relacionadas à matemática, abordando tópicos como **funções**, **polinômios**, **operações básicas** e **conjuntos**. O sistema utiliza a LLM **Gemini** para gerar respostas com base em um banco de dados criado a partir de livros em PDF.

---

## Funcionalidades

- **Processamento de PDFs**: O sistema extrai texto de livros em PDF e os divide em chunks processáveis.
- **Banco de Dados de Chunks**: Os chunks são armazenados e indexados utilizando **FAISS** para busca eficiente.
- **Respostas Otimizadas**: Perguntas do usuário são respondidas usando a LLM Gemini, com suporte a avaliação da qualidade das respostas.
- **Foco em Matemática**: Respostas personalizadas para tópicos específicos da matemática.

---

## Pré-requisitos

Antes de começar, certifique-se de ter instalado:

- **Python 3.9+**
- Bibliotecas Python:
  - `torch`
  - `transformers`
  - `faiss`
  - `pdfplumber`
  - `nltk`
  - `google-generativeai`

Instale as dependências com:
```bash
pip install -r requirements.txt

Estrutura geral do projeto:

├── pdf_process.py        # Script para processar PDFs e criar o banco de dados
├── pipeline_final.py     # Script principal do chatbot RAG
├── requirements.txt      # Dependências do projeto
├── embeddings_index.faiss# Índice FAISS para busca
├── documents_ids.pkl     # Banco de dados de chunks e IDs
├── README.md             # Documentação do projeto
