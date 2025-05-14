import streamlit as st
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAI # Corrected import
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Set up the Streamlit page
st.set_page_config(page_title="Chat with a Variable", page_icon=":book:")
st.title("Chat with a Variable")


# Define the string containing the details
details = """
Damian Montero is a 52 year old developer and AI fanatic with a teen aged daughter and a wife.
He's been married for 25 years and has a passion for technology.
This morning he had a bagel and a coffee for breakfast but currently he's in love with everything to do with AI.
"""

# Create a LangChain document from the string
document = Document(page_content=details)

# Load the document into a vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")  # Use Google Generative AI embeddings
vectorstore = FAISS.from_documents([document], embeddings)

# Initialize the LLM and QA chain
llm = GoogleGenerativeAI(model="models/gemini-2.5-flash-preview-04-17")  # Use Google Generative AI LLM
chain = load_qa_chain(llm, chain_type="stuff")

# Accept user queries
query = st.text_input("Ask a question about the details:")

if query:
    # Perform the query














    docs = vectorstore.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)
    st.write(f"**Answer:** {response}")