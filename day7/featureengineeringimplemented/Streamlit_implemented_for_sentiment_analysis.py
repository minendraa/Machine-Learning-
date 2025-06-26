import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier

# Load the model and vectorizer
@st.cache_resource
def load_components():
    vectorizer = joblib.load("newvectorizerforreview.pkl")
    model = joblib.load("newmodelforperceptronafterfeatureengineering.pkl")
    return vectorizer, model

vectorizer, model = load_components()

# Streamlit app
st.title("🎵 Musical Instrument Reviews Sentiment Analysis")
st.write("This app predicts whether a review is Positive, Negative, or Neutral")

# User input
review_text = st.text_area("Enter your review:", 
                          "This product is amazing! The sound quality exceeded my expectations.")

# Prediction function
def predict_sentiment(text):
    # Vectorize the text
    text_vec = vectorizer.transform([text])
    # Make prediction
    prediction = model.predict(text_vec)
    return prediction[0]

# Make and display prediction
if st.button("Analyze Sentiment"):
    if review_text.strip() == "":
        st.warning("Please enter a review")
    else:
        sentiment = predict_sentiment(review_text)
        
        # Display result with emoji
        if sentiment == "Positive":
            st.success(f"Predicted Sentiment: {sentiment} 😊")
        elif sentiment == "Negative":
            st.error(f"Predicted Sentiment: {sentiment} 😞")
        else:
            st.info(f"Predicted Sentiment: {sentiment} 😐")

# Optional: Add some sample reviews
st.sidebar.header("Sample Reviews")
sample_reviews = [
    "This guitar is perfect! The sound is rich and full.",
    "Not worth the money. Broke after 2 weeks of use.",
    "It's okay, but I expected better quality for the price."
]

for i, review in enumerate(sample_reviews):
    if st.sidebar.button(f"Sample {i+1}"):
        review_text = review

# Display the current review being analyzed
st.subheader("Review Preview")
st.write(review_text)