import streamlit as st
import google.generativeai as genai
google_api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=google_api_key)

# Set up the Streamlit page
st.set_page_config(page_title="Simple Gemini Chatbot", page_icon=": разговаривающий_облачко :") #Corrected emoji
st.title("Simple Chatbot with Google")


# Initialize the Gemini model
model = genai.GenerativeModel("models/gemini-2.5-flash-preview-04-17")  # Use gemini-pro for chat

# Initialize chat session state
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])

# Display chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
prompt = st.chat_input("What can I help you with?")

if prompt:
    # Add user message to chat history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from Gemini
    try:
        response = st.session_state.chat_session.send_message(prompt)
        # response = model.generate_content(prompt) # Removed
        # Add assistant message to chat history and display
        st.session_state.messages.append({"role": "assistant", "content": response.text})
        with st.chat_message("assistant"):
            st.markdown(response.text)
    except Exception as e:
        st.error(f"An error occurred: {e}")
