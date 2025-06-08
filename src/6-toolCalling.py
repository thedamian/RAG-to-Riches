import streamlit as st
import google.generativeai as genai
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# Configure Google Gemini API
google_api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=google_api_key)

#Open AI API key for LangChain
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

#Llama API key for LangChain
LLAMA_API_KEY = st.secrets["LLAMA_API_KEY"]

# Define function to return hardcoded temperature
def getWeatherByCity(city):
    print (f"\nReceived tooling request for weather in {city}")
    return f"The temperature in {city} is 72°F."

# Wrap function in LangChain's Tool wrapper
weather_tool = Tool(
    name="getWeatherByCity",
    func=getWeatherByCity,
    description="Returns the temperature for a given city."
)

# Google Gemini LLM configuration
#llm = ChatOpenAI(openai_api_base="https://generativelanguage.googleapis.com/v1beta/", openai_api_key=google_api_key, model_name="gemini-2.5-flash-preview-05-20")

# Open AI LLM configuration
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4.1")

# Meta's Llama 4 LLM configuration
#llm = ChatOpenAI(openai_api_base="https://api.llama.com/v1/",openai_api_key=LLAMA_API_KEY, model_name="Llama-4-Maverick-17B-128E-Instruct-FP8")

# Initialize LangChain agent with the weather tool
agent = initialize_agent(
    tools=[weather_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True
)

# Streamlit UI setup
st.set_page_config(page_title="Gemini Chatbot with Tooling", page_icon=":cloud_talking:")
st.title("Chatbot with Google Gemini & LangChain Tooling")

# Chat session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input box
user_input = st.chat_input("Ask me anything!")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Agent dynamically handles queries and tools
    response = agent.run(user_input)

    # Display response
    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
