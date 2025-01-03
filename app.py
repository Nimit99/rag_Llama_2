import streamlit as st
import PyPDF2
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline

def main():
    st.title("Simple RAG with Streamlit")

    # 1) File Uploader
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file is not None:
        # 2) Extract text from PDF
        pdf_text = []
        reader = PyPDF2.PdfReader(uploaded_file)
        for page_idx in range(len(reader.pages)):
            page_text = reader.pages[page_idx].extract_text()
            pdf_text.append(page_text)
        full_text = " ".join(pdf_text)

        # 3) Chunk the text
        chunk_size = 500
        chunks = [full_text[i : i + chunk_size] for i in range(0, len(full_text), chunk_size)]

        # 4) Create embeddings + build Faiss index
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True)
        embedding_dim = chunk_embeddings.shape[1]
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(chunk_embeddings.cpu().numpy())

        # Retrieval function
        def retrieve_top_k(query, k=2):
            query_emb = embedder.encode([query], convert_to_tensor=True)
            distances, indices = index.search(query_emb.cpu().numpy(), k)
            return [chunks[i] for i in indices[0]]

        # 5) Load Flan-T5 model (text2text-generation pipeline)
        generator = pipeline("text2text-generation", model="google/flan-t5-base")

        def rag_answer(query):
            top_chunks = retrieve_top_k(query, k=2)
            context_text = " ".join(top_chunks)
            prompt = f"Context: {context_text}\n\nQuery: {query}\nAnswer:"
            response = generator(prompt, max_length=128)[0]["generated_text"]
            return response

        # 6) User query + Generate answer
        user_query = st.text_input("Enter your question here:")
        if st.button("Get Answer") and user_query:
            answer = rag_answer(user_query)
            st.write("**Answer:** ", answer)

if __name__ == "__main__":
    main()
