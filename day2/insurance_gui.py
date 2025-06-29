import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load and preprocess dataset
df = pd.read_csv('insurance.csv')

# Label Encoding
le_smoker = LabelEncoder()
df['smoker'] = le_smoker.fit_transform(df['smoker'])

le_region = LabelEncoder()
df['region'] = le_region.fit_transform(df['region'])

le_sex = LabelEncoder()
df['sex'] = le_sex.fit_transform(df['sex'])

# Features and target
X = df[['smoker', 'age', 'region', 'bmi', 'children']]
y = df['charges']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit app
st.title("Insurance Charges Prediction App")

st.markdown("Enter the following details to predict the insurance charges:")

# User inputs
age = st.slider("Age", 18, 100, 30)
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
smoker = st.selectbox("Smoker", ['no', 'yes'])
region = st.selectbox("Region", le_region.classes_)  # Display original region labels

# Convert inputs for prediction
smoker_encoded = le_smoker.transform([smoker])[0]
region_encoded = le_region.transform([region])[0]

# Predict
if st.button("Predict Charges"):
    input_data = np.array([[smoker_encoded, age, region_encoded, bmi, children]])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Insurance Charges: ${prediction:,.2f}")

    # Optional: Model performance on test set
    y_pred = model.predict(X_test)
    st.markdown("### Model Performance (on test data):")
    st.write(f"Mean Absolute Error: ${mean_absolute_error(y_test, y_pred):,.2f}")
    st.write(f"Mean Squared Error: ${mean_squared_error(y_test, y_pred):,.2f}")
    st.write(f"R-squared Score: {r2_score(y_test, y_pred):.2f}")
