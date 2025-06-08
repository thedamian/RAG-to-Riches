import streamlit as st
import sqlite3
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
GOOGLE_GENAI_API_KEY = st.secrets["GOOGLE_API_KEY"]


# Initialize LangChain with Google Gemini AI
llm = GoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", google_api_key=GOOGLE_GENAI_API_KEY)

# Connect to database and retrieve player info
def get_players():
    conn = sqlite3.connect("players.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name, hits FROM topPlayers")
    players = cursor.fetchall()
    conn.close()
    
    docs = [Document(page_content=f"Player: {name}, Hits: {hits}") for name, hits in players]
    return docs

# Initialize FAISS for Retrieval-Augmented Generation (RAG)
def setup_rag():
    docs = get_players()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)  # Embedding for search
    return vectorstore

vectorstore = setup_rag()

# Streamlit UI
st.title("🏆 Baseball Player Stats (RAG-Powered)")
user_query = st.text_input("Ask about the top players:")

if user_query:
    retriever = vectorstore.as_retriever()
    context = retriever.get_relevant_documents(user_query)
    
    response = llm(
        f"Based on the database records, {context}. Answer concisely."
    )
    
    st.write("🔍 **AI Response:**")
    st.write(response)
