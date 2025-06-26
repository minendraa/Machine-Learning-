import pandas as pd
import joblib
from tkinter import *
from sklearn.feature_extraction.text import CountVectorizer

# Load the saved model and vectorizer
model = joblib.load("modelforemailspamdetection.pkl")
vectorizer = joblib.load("vectorizer.pkl")  # Make sure you saved this too!

# Function to predict spam or ham
def predict_spam():
    message = entry.get("1.0", END).strip()
    if not message:
        result_label.config(text="Please enter a message.")
        return
    
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)
    
    if prediction[0] == 1:
        result_label.config(text="Result: Spam", fg="red")
    else:
        result_label.config(text="Result: Ham", fg="green")

# GUI setup
root = Tk()
root.title("Email Spam Detector")
root.geometry("400x300")
root.config(bg="#f0f0f0")

Label(root, text="Enter your email message:", bg="#f0f0f0", font=("Arial", 12)).pack(pady=10)

entry = Text(root, height=6, width=45, font=("Arial", 10))
entry.pack()

Button(root, text="Predict", command=predict_spam, bg="#4caf50", fg="white", font=("Arial", 11)).pack(pady=10)

result_label = Label(root, text="", font=("Arial", 14), bg="#f0f0f0")
result_label.pack()

root.mainloop()