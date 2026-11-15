import streamlit as st
import pickle

st.title("AI Fake News Detection")

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

news = st.text_area("Enter News Article")

if st.button("Detect News"):
    if news.strip():
        text = vectorizer.transform([news])
        result = model.predict(text)[0]

        if result == 0:
            st.error("Fake News")
        else:
            st.success("Real News")
    else:
        st.warning("Please enter news")
