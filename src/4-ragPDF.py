import streamlit as st
import fitz  # PyMuPDF for PDF parsing
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI 
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
#from langchain.llms import GooglePalm  # Using Gemini through LangChain
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def extract_text_from_pdf(pdf_file):
    """Extracts text from uploaded PDF file."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join([page.get_text("text") for page in doc])
    return text

def process_document(text):
    """Splits text into chunks and stores in a FAISS vector store."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    documents = [Document(page_content=t) for t in texts]

    # Embedding model for vector storage
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
    vector_store = FAISS.from_documents(documents, embeddings)

    return vector_store

def chat_with_pdf(query, vector_store):
    """Uses LangChain with Gemini API for RAG-based response."""
    retriever = vector_store.as_retriever()
    relevant_docs = retriever.get_relevant_documents(query)
    llm = GoogleGenerativeAI(model="models/gemini-2.5-flash-preview-04-17")
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    response = chain.run(input_documents=relevant_docs, question=query)

    return response

# Streamlit UI
st.title("Chat with a PDF")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    st.success("PDF uploaded successfully!")
    text = extract_text_from_pdf(uploaded_file)
    vector_store = process_document(text)

    user_input = st.text_input("Ask something about the PDF:")
    if user_input:
        response = chat_with_pdf(user_input, vector_store)
        st.write("Response:", response)
