import streamlit as st
from openai import OpenAI
import os
from PyPDF2 import PdfReader

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

if 'openai_client' not in st.session_state:
    api_key = st.secrets['key1']
    st.session_state.openai_client = OpenAI(api_key=api_key)

def coll_function():
    client = chromadb.PersistentClient()
    collection = client.get_or_create_collection("L4_Collection", metadata={"hnsw:space": "cosine", "hnsw:M": 32})
    
    datafiles_path = os.path.join(os.getcwd(), "datafiles")
    pdf_files = [f for f in os.listdir(datafiles_path) if f.endswith('.pdf')]
        
    for pdf_file in pdf_files:
        file_path = os.path.join(datafiles_path, pdf_file)
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
                
            st.write(f"Extracted text from {pdf_file}: {text[:200]}")  # Log extracted text
            openai_client = st.session_state.openai_client
            
            try:
                response = openai_client.embeddings.create(
                    input=text,
                    model="text-embedding-3-small"
                )
                embedding = response.data[0].embedding
            except Exception as e:
                st.error(f"Error generating embedding for {pdf_file}: {e}")
                continue
            
            collection.add(
                documents=[text],
                ids=[pdf_file],
                embeddings=[embedding]
            )
    
    st.session_state.l4_collection = collection

if st.button("Setup VectorDB"):
    coll_function()

user_input = st.text_input("Enter some text:")
if user_input:
    openai_client = st.session_state.openai_client
    collection = st.session_state.l4_collection
    
    try:
        response = openai_client.embeddings.create(
            input=user_input,
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        
        # Query the collection for similar documents
        results = collection.query(
            query_embeddings=[embedding],
            n_results=3,
            include=['distances']
        )
        
        # Combine results and sort them by distances
        combined_results = list(zip(results['ids'], results['distances']))
        combined_results.sort(key=lambda x: x[1])  # Sort by distance
        
        # Display the sorted results
        for doc_id, distance in combined_results:
            st.write(f"The following file/syllabus might be helpful: {doc_id}")
            st.write(f"The distance: {distance}")
    
    except Exception as e:
        st.error(f"Error generating embedding for user input: {e}")
