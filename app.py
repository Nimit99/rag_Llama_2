import streamlit as st
import os
import docx2txt
import csv
import json
import PyPDF2
import sentence_transformers
import transformers

from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

from transformers import pipeline

# ---------- 1. Helper Functions ---------- #

def parse_pdf(file):
    """Extract text from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text() or ""
    return text

def parse_docx(file):
    """Extract text from a Word file."""
    return docx2txt.process(file)

def parse_csv(file):
    """Extract text from a CSV file by reading row by row."""
    text_data = []
    import io
    csv_file = io.TextIOWrapper(file, encoding='utf-8')
    reader = csv.reader(csv_file)
    for row in reader:
        text_data.append(" ".join(row))
    return "\n".join(text_data)

def parse_json(file):
    """Extract text from a JSON file by flattening it into text."""
    import io
    json_file = json.load(io.TextIOWrapper(file, encoding='utf-8'))
    return str(json_file)


def create_vector_store(text_chunks):
    """Create Chroma vectorstore from text chunks."""
    docs = [Document(page_content=chunk) for chunk in text_chunks]
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(docs, embedding=embedder)
    return vector_store

def create_vector_store(text_chunks):
    docs = [Document(page_content=chunk) for chunk in text_chunks]
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embedder)
    return vector_store


# ---------- 2. Streamlit App ---------- #

def main():
    st.title("RAG App with Llama (or Other HF Model)")
    
    # 1. File uploader
    uploaded_file = st.file_uploader(
        "Upload a PDF, DOCX, CSV, or JSON file", 
        type=["pdf", "docx", "csv", "json"]
    )
    
    if uploaded_file is not None:
        # 2. Parse file
        file_type = uploaded_file.name.split(".")[-1].lower()
        raw_text = ""
        
        if file_type == "pdf":
            raw_text = parse_pdf(uploaded_file)
        elif file_type == "docx":
            raw_text = parse_docx(uploaded_file)
        elif file_type == "csv":
            raw_text = parse_csv(uploaded_file)
        elif file_type == "json":
            raw_text = parse_json(uploaded_file)
        else:
            st.error("Unsupported file type.")
            return

        # 3. Split text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(raw_text)

        # 4. Create vector store
        vector_store = create_vector_store(chunks)

        # 5. Load Hugging Face LLM (for example, Llama 2)
        #    Make sure you have accepted the license on Hugging Face:
        #    https://huggingface.co/meta-llama/Llama-2-7b-hf
        model_name = "meta-llama/Llama-2-7b-hf"  # or any other HF model
        hf_pipe = pipeline(
            "text-generation", 
            model=model_name, 
            torch_dtype="auto",  # might use torch.float16 if GPU available
            device_map="auto"
        )
        llm = HuggingFacePipeline(pipeline=hf_pipe)
        
        # 6. Create a RetrievalQA chain
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )

        # 7. Ask your question
        question = st.text_input("Ask a question about your uploaded document:")
        
        if question:
            answer = qa_chain.run(question)
            st.write("**Answer**:", answer)
            
if __name__ == "__main__":
    main()
