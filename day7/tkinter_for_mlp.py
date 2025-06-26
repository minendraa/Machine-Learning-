import tkinter as tk
from tkinter import messagebox
import joblib

# Load trained model and vectorizer
model = joblib.load("modelforperceptron.pkl")
vectorizer = joblib.load("vectorizerforreview.pkl")

# Prediction function
def predict_rating():
    review = review_input.get("1.0", tk.END).strip()
    if not review:
        messagebox.showwarning("Input Error", "Please enter a review.")
        return

    transformed = vectorizer.transform([review])
    prediction = model.predict(transformed)

    result_label.config(
        text=f"Predicted Rating: ⭐ {prediction[0]} / 5", fg="green", font=("Helvetica", 14, "bold")
    )

# GUI Setup
root = tk.Tk()
root.title("Amazon Review Rating Predictor")
root.geometry("450x300")
root.resizable(False, False)

title_label = tk.Label(root, text="🎵 Review Rating Predictor 🎵", font=("Helvetica", 16, "bold"))
title_label.pack(pady=10)

input_label = tk.Label(root, text="Enter Review Text:")
input_label.pack()

review_input = tk.Text(root, height=5, width=50)
review_input.pack(pady=5)

predict_button = tk.Button(root, text="Predict Rating", command=predict_rating, bg="#2980b9", fg="white", font=("Arial", 12))
predict_button.pack(pady=10)

result_label = tk.Label(root, text="Predicted Rating: ", font=("Arial", 12))
result_label.pack(pady=10)

root.mainloop()
