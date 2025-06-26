import tkinter as tk
from tkinter import messagebox
import joblib

# Load the model and vectorizer
model = joblib.load("modelforperceptron.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Function to predict spam or ham
def predict_spam():
    msg = entry.get("1.0", tk.END).strip()
    if not msg:
        messagebox.showwarning("Input Error", "Please enter a message.")
        return
    
    transformed_msg = vectorizer.transform([msg])
    prediction = model.predict(transformed_msg)

    if prediction[0] == 1:
        result_label.config(text="Result: 🚫 SPAM", fg="red")
    else:
        result_label.config(text="Result: ✅ HAM (Not Spam)", fg="green")

# Create the GUI
root = tk.Tk()
root.title("Perceptron Spam Detector")
root.geometry("400x300")
root.resizable(False, False)

title_label = tk.Label(root, text="Spam Message Classifier", font=("Helvetica", 16, "bold"))
title_label.pack(pady=10)

entry_label = tk.Label(root, text="Enter your message:")
entry_label.pack()

entry = tk.Text(root, height=6, width=40)
entry.pack(pady=5)

check_button = tk.Button(root, text="Check Spam", command=predict_spam, bg="#3498db", fg="white", font=("Arial", 12))
check_button.pack(pady=10)

result_label = tk.Label(root, text="Result: ", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
