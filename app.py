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

# ---------- HELPER FUNCTIONS ---------- #

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
    """Extract text from a DOCX file."""
    return docx2txt.process(file).strip()

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
        # Join each row's cells with a space
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

def create_vector_store(text_chunks):
    """
    Create a FAISS vectorstore from text chunks using HuggingFaceEmbeddings.
    We check if the chunks list is empty; if it is, we return None or raise an error.
    """
    if not text_chunks:
        return None  # or raise ValueError("No text chunks to embed.")

    # Convert chunks to Document objects
    docs = [Document(page_content=chunk) for chunk in text_chunks]

    # Use the community-based HuggingFaceEmbeddings
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS vector store from documents
    vector_store = FAISS.from_documents(docs, embedder)
    return vector_store

# ---------- MAIN APP ---------- #

def main():
    st.title("RAG App with FAISS + Hugging Face")

    uploaded_file = st.file_uploader(
        "Upload a PDF, DOCX, CSV, or JSON file",
        type=["pdf", "docx", "csv", "json"]
    )

    if uploaded_file is not None:
        # Figure out file extension
        file_ext = uploaded_file.name.split(".")[-1].lower()
        raw_text = ""

        # 1) Parse file based on extension
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

        # If the file is empty or parsing failed
        if not raw_text.strip():
            st.error("No text found in the document. Please check your file.")
            return

        # 2) Split text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(raw_text)

        if not chunks:
            st.error("The document text was parsed, but no chunks were created.")
            return

        # 3) Create Vector Store
        vector_store = create_vector_store(chunks)
        if not vector_store:
            st.error("Vector store could not be created (empty text or embedding error).")
            return

        # 4) Load a Hugging Face LLM
        # Example: "meta-llama/Llama-2-7b-hf" if you have accepted the license
        # You can also pick a smaller model if you are limited on GPU.
        model_name = "meta-llama/Llama-2-7b-hf"
        hf_pipe = pipeline(
            "text-generation",
            model=model_name,
            # For large models, consider torch_dtype=torch.float16 if you have GPU
            device_map="auto"  
        )
        llm = HuggingFacePipeline(pipeline=hf_pipe)

        # 5) Build a RetrievalQA chain
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )

        # 6) User question
        question = st.text_input("Ask a question about the uploaded document:")

        if question:
            with st.spinner("Generating answer..."):
                answer = qa_chain.run(question)
            st.write("**Answer**:", answer)


if __name__ == "__main__":
    main()
