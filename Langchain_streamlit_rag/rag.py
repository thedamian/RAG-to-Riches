import streamlit as st
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
import google.generativeai as genai

# Set your Google API key
google_api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=google_api_key)

def load_and_index_website(url):
    """Loads a website, splits it into chunks, and creates a FAISS index."""
    loader = WebBaseLoader(url)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") # or other embedding models
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

def create_conversational_chain(vectorstore):
    """Creates a conversational retrieval chain using Gemini Flash."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", convert_system_message_to_human=True) #Use gemini flash
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return chain

def main():
    st.title("Website Chatbot (Gemini Flash)")

    url = st.text_input("Enter the website URL:")

    if url:
        try:
            with st.spinner("Loading and indexing website..."):
                vectorstore = load_and_index_website(url)
                st.session_state.chain = create_conversational_chain(vectorstore)
            st.success("Website loaded and indexed successfully!")

            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("Ask a question about the website:"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    response = st.session_state.chain({"question": prompt})
                    full_response = response["answer"]
                    message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

