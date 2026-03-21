⚖️ AI-Based Legal Document Assistant

An intelligent Q&A system designed to analyze legal PDFs and provide citation-based answers. This project uses RAG (Retrieval-Augmented Generation) to ensure accuracy and prevent hallucinations by grounding the AI's responses strictly in the provided document text.

🚀 Live Demo

Click here to view the Live App (https://ai-legal-assistant-qm.streamlit.app/)

🛠️ System Methodology (RAG Workflow)

This project implements a professional AI pipeline to handle unstructured legal data:


Ingestion: Extracts text and page metadata from PDF files.

Chunking: Breaks long legal text into 1000-character segments with a 200-character overlap to preserve context.

Embedding: Converts text into high-dimensional vectors using sentence-transformers/all-MiniLM-L6-v2.

Vector Storage: Stores and indexes data in FAISS for fast semantic search.

Retrieval: Finds the top 5 most relevant legal segments based on a user’s question.

Augmentation & Generation: Feeds the relevant context into Llama 3.3 70B via Groq to generate a precise, cited answer.

🏗️ Tech Stack

Framework: Streamlit (Frontend & Deployment)

AI Model: Llama 3.3 70B via Groq API

Orchestration: LangChain

Vector DB: FAISS

Embeddings: HuggingFace Community Transformers

📦 Installation

To run this project locally, follow these steps:

Clone the Repo:

git clone https://github.com/mekha-00/ai-legal-assistant.git

cd ai-legal-assistant

Install Dependencies:

pip install -r requirements.txt

Configure Secrets:

Create a folder .streamlit and a file secrets.toml inside it:

GROQ_API_KEY = "your_groq_api_key_here"

Run the Application:

streamlit run app.py

📜 Requirements

Your requirements.txt should include:
streamlit,
groq,
langchain,
langchain-community,
pypdf,
sentence-transformers,
faiss-cpu,

🛡️ Features
Strict Grounding: The AI only answers based on the uploaded document.

Page Citations: Every answer includes the exact page number from the PDF.

High Speed: Utilizes Groq’s LPU for near-instant inference speeds.

Safe Handling: Uses Streamlit Secrets for API key protection.
