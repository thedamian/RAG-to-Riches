import requests
import json
import streamlit as st

st.title(" ☕☕ List of coffees ☕☕ ")
type_of_coffee = st.selectbox("type of coffee?",["hot","iced"])
filtertitle = st.text_input("filter?")

if type_of_coffee:
    url = "https://api.sampleapis.com/coffee/" + type_of_coffee + "/?title_like=" + filtertitle
    st.text(f"Fetching: {url} ")
    response = requests.get(url)
    coffees =  json.loads(response.content)
    #st.table(coffees)
    for coffee in coffees:
        st.text(coffee["title"])
        st.image(coffee["image"])