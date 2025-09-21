pip install PyMuPDF
pip install faiss-cpu
pip install groq

#Importing required libraries
import fitz
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import numpy as np

#Load embedding model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

#Groq API setup
LLAMA3_1_GROQ_API_KEY = "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"
LLAMA3_1_MODEL_NAME = "llama-3.3-70b-versatile"
client = Groq(api_key=LLAMA3_1_GROQ_API_KEY)

#function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text
 
#function to create FAISS index
def create_faiss_index(text_chunks):
    embeddings = embed_model.encode(text_chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings, text_chunks
 
#Function to retrieve relevant chunks
def retrieve_relevant_chunks(query, index, embeddings, text_chunks, k=3):
    query_embedding = embed_model.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_embedding, k)
    return [text_chunks[i] for i in indices[0]]

#Function to query Llama3 with context
def query_llama3(context, question):
    response = client.chat.completions.create(
        model=LLAMA3_1_MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an expert at answering questions based provided documents."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"} 
],
 temperature=0,
 max_tokens=8000,
 top_p=1
 )
 return response.choices[0].message.content
 
#Example Usage
pdf_path = "/content/data (1).pdf"
question = "What is supervised learning?"

#Extract text from PDF
text = extract_text_from_pdf(pdf_path)
chunks = text.split(". ")

#Create FAISS index
index, embeddings, text_chunks = create_faiss_index(chunks)

#Retrieve relevant text
retrieved_chunks = retrieve_relevant_chunks(question, index, embeddings, text_chunks)
retrieved_text = " ".join(retrieved_chunks)

#Query Llama3
answer = query_llama3(retrieved_text, question)
print("Answer:", answer)
