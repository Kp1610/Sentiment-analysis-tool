import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")


labels = ["very negative", "negative", "neutral", "positive", "very positive"]
emojis = {
    "very negative": "😞",
    "negative": "😐",
    "neutral": "😐",
    "positive": "😊",
    "very positive": "😁"
}

page = st.sidebar.radio("Navigation", ["Multilingual Sentiment Analysis", "About Me"])


if page == "Multilingual Sentiment Analysis":
    st.title("Multilingual Sentiment Analysis")
    input_text = st.text_area("Enter text for sentiment analysis", height=200)

    if st.button("Analyze Sentiment"):
        if input_text:
            inputs = tokenizer(input_text, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            

            scores = outputs.logits[0].softmax(dim=0)
            sentiment = labels[scores.argmax()]

            st.write(f"Sentiment: {sentiment} {emojis[sentiment]}")
            st.write("Confidence Scores:")
            for label, score in zip(labels, scores):
                st.write(f"{label.capitalize()}: {score:.2f}")
        else:
            st.warning("Please enter text for analysis")


elif page == "About Me":
    st.title("About Me")
    st.image("Krutarth_Pandya_PassportSize_Photo.jpg", caption="Krutarth Pandya", width=300 )
    st.markdown("I am a Computer Science Engineering student in my fourth year at Karnavati University.")
    st.markdown("I have a keen interest in data science and machine learning.")

