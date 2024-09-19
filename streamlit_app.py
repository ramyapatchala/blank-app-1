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
    st.session_state.openai_client = OpenAI(api_key = api_key)

def coll_function():
    client = chromadb.PersistentClient()
    client.delete_collection("L4_Collection")
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
            openai_client = st.session_state.openai_client
            response = openai_client.embeddings.create(
                        input=text,
                        model="text-embedding-3-small")
            embedding = response.data[0].embedding
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
    response = openai_client.embeddings.create(
        input=user_input,
        model="text-embedding-3-small"  # Use the correct model name
        )
    embedding = response.data[0].embedding
        # Query the collection for similar documents
    results = collection.query(
            query_embeddings=[embedding],
            n_results=3
        )
        # Display the results
    for i in range(len(results['ids'])):
        doc_id = results['ids'][i]
        st.write(f"The following file/syllabus might be helpful: {doc_id}")

    
    

    
