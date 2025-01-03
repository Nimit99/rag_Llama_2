import streamlit as st
import os
import docx2txt
import csv
import json
import PyPDF2

# For embedding + LLM pipeline
import sentence_transformers
import transformers
from transformers import pipeline

# LangChain / Community imports
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

# ---------------------------------------------------------------------
# 1) CONSTANTS / TOKEN
# ---------------------------------------------------------------------
# NOTE: For real production, do NOT hardcode tokens.
# Put them in secrets or environment variables.
HF_TOKEN = "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"  # <-- your token

# ---------------------------------------------------------------------
# 2) PARSING FUNCTIONS
# ---------------------------------------------------------------------
def parse_pdf(file):
    """Extract text from a PDF file using PyPDF2."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page_text = pdf_reader.pages[page_num].extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

def parse_docx(file):
    """Extract text from a DOCX file using docx2txt."""
    text = docx2txt.process(file)
    return text.strip() if text else ""

def parse_csv(file):
    """
    Extract text from a CSV file by reading row by row.
    Joins the cells in each row with spaces, then joins rows with newlines.
    """
    import io
    text_data = []
    csv_file = io.TextIOWrapper(file, encoding='utf-8')
    reader = csv.reader(csv_file)
    for row in reader:
        text_data.append(" ".join(row))
    return "\n".join(text_data).strip()

def parse_json(file):
    """
    Extract text from a JSON file by flattening it into text.
    This is a simplistic approach: it just str() everything.
    """
    import io
    json_file = json.load(io.TextIOWrapper(file, encoding='utf-8'))
    return str(json_file).strip()

# ---------------------------------------------------------------------
# 3) CREATE VECTOR STORE (FAISS)
# ---------------------------------------------------------------------
def create_vector_store(text_chunks):
    """
    Create a FAISS vectorstore from text chunks using HuggingFaceEmbeddings.
    If chunks are empty, return None.
    """
    if not text_chunks:
        return None

    docs = [Document(page_content=chunk) for chunk in text_chunks]
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embedder)
    return vector_store

# ---------------------------------------------------------------------
# 4) STREAMLIT APP
# ---------------------------------------------------------------------
def main():
    st.title("RAG with Llama 2 & FAISS")

    # 1) File Upload
    uploaded_file = st.file_uploader(
        "Upload a PDF, DOCX, CSV, or JSON file",
        type=["pdf", "docx", "csv", "json"]
    )

    if uploaded_file is not None:
        # 2) Parse the file by extension
        file_ext = uploaded_file.name.split(".")[-1].lower()
        raw_text = ""

        if file_ext == "pdf":
            raw_text = parse_pdf(uploaded_file)
        elif file_ext == "docx":
            raw_text = parse_docx(uploaded_file)
        elif file_ext == "csv":
            raw_text = parse_csv(uploaded_file)
        elif file_ext == "json":
            raw_text = parse_json(uploaded_file)
        else:
            st.error("Unsupported file type.")
            return

        # Check if parsed text is empty
        if not raw_text.strip():
            st.error("No text found in the document. Please check your file.")
            return

        # 3) Split text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(raw_text)

        if not chunks:
            st.error("Document was parsed, but no text chunks were created.")
            return

        # 4) Create vector store
        vector_store = create_vector_store(chunks)
        if not vector_store:
            st.error("Vector store could not be created (empty text or error).")
            return

        # 5) Load Llama 2 model via pipeline
        # You must have accepted the Llama 2 license on Hugging Face
        # (https://huggingface.co/meta-llama).
        # Also we pass your token so it can access the gated model.
        model_name = "meta-llama/Llama-2-7b-hf"
        hf_pipe = pipeline(
            "text-generation",
            model=model_name,
            device_map="auto",
            use_auth_token=HF_TOKEN  # Pass the token
        )

        llm = HuggingFacePipeline(pipeline=hf_pipe)

        # 6) Build a RetrievalQA chain
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )

        # 7) Ask your question
        question = st.text_input("Ask a question about the document:")
        if question:
            with st.spinner("Generating answer..."):
                answer = qa_chain.run(question)
            st.write("**Answer**:", answer)

if __name__ == "__main__":
    main()
