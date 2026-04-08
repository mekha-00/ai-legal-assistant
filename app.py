import os
import tempfile
import streamlit as st

from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -----------------------------
# Streamlit Setup
# -----------------------------

st.set_page_config(page_title="AI Legal Document Assistant")

st.title("⚖️ AI Legal Document Assistant")
st.write("Upload a legal PDF and ask questions with citation-based answers.")

# -----------------------------
# API Key
# -----------------------------

api_key = st.secrets.get("GROQ_API_KEY")


if not api_key:
    st.error("API key not found. Set GROQ_API_KEY environment variable.")
    st.stop()

client = Groq(api_key=api_key)

# -----------------------------
# Process PDF
# -----------------------------

def process_pdf(uploaded_file):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    if not documents:
        st.error("PDF could not be read")
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    if len(chunks) == 0:
        st.error("No readable text found in PDF")
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.from_documents(chunks, embeddings)

    return vector_db


# -----------------------------
# File Upload
# -----------------------------

uploaded_file = st.file_uploader("Upload a Legal PDF", type="pdf")

if uploaded_file:

    st.success("PDF uploaded successfully")

    with st.spinner("Processing document..."):
        vector_db = process_pdf(uploaded_file)

    if vector_db is None:
        st.stop()

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    st.success("Document processed. You can ask questions now.")

    # -----------------------------
    # Ask Question
    # -----------------------------

    query = st.text_input("Ask a question from the document")

    if query:

        docs = retriever.invoke(query)

        context = "\n\n".join([
            f"(Page {doc.metadata.get('page') + 1}) {doc.page_content}"
            for doc in docs
        ])

        # 🔥 EXACT PROMPT (DEEPER EXPLANATION)
        prompt = f"""
You are a legal document assistant.

STRICT RULES:
- Answer ONLY using the provided context
- DO NOT use outside knowledge
- If answer is not in the document, say: "Not found in the document"
- Always include page numbers in your answer

EXPLANATION REQUIREMENTS:
- Provide a detailed explanation in simple language (minimum 2–4 sentences)
- Break down complex legal text into clear meaning
- Explain what it means and why it is important
- Do NOT copy text as explanation
- The explanation must be based ONLY on the provided context

FORMAT:

Answer:
<Detailed explanation (2–4 sentences, simple and clear)>

Supporting Evidence:
- (Page X): <relevant extracted text>

Context:
{context}

Question:
{query}
"""

        # -----------------------------
        # GROQ API CALL
        # -----------------------------

        with st.spinner("Analyzing document..."):
            response = client.chat.completions.create(
                model="model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}]
            )

            answer = response.choices[0].message.content

        # -----------------------------
        # Display Answer
        # -----------------------------
        
        st.markdown(answer)
