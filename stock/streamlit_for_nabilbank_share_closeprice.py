#Streamlit implementation on nabil bank share close price prediction system
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Set page title and layout
st.set_page_config(page_title="Nabil Bank Stock Analyzer", layout="wide")

# Title
st.title("Nabil Bank Stock Price Analysis and Prediction")

# Load data directly
try:
    df = pd.read_csv('NabilBank.csv')
    
    # Show raw data
    st.subheader("Stock Data")
    st.write(df.head())
    
    # Visualization section
    st.subheader("Price Trends")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['High'], label='High', color='red')
    ax.plot(df['Low'], label='Low', color='blue')
    ax.set_title('Daily High and Low Prices')
    ax.set_xlabel('Days')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    # Model training section
    st.subheader("Price Prediction Model")
    
    features = ['Open', 'High', 'Low']
    X = df[features]
    y = df['Close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Display model coefficients
    st.write("**Model Coefficients:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Open Coeff", f"{model.coef_[0]:.4f}")
    with col2:
        st.metric("High Coeff", f"{model.coef_[1]:.4f}")
    with col3:
        st.metric("Low Coeff", f"{model.coef_[2]:.4f}")
    with col4:
        st.metric("Intercept", f"{model.intercept_:.4f}")
    
    # Model evaluation
    y_pred = model.predict(X_test)
    st.write("**Model Performance:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MAE", f"{metrics.mean_absolute_error(y_test, y_pred):.4f}")
    with col2:
        st.metric("MSE", f"{metrics.mean_squared_error(y_test, y_pred):.4f}")
    with col3:
        st.metric("R-squared", f"{metrics.r2_score(y_test, y_pred):.4f}")
    
    # Prediction section
    st.subheader("Predict Closing Price")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            open_price = st.number_input("Open Price", min_value=0.0, value=float(df['Open'].iloc[-1]))
        with col2:
            high_price = st.number_input("High Price", min_value=0.0, value=float(df['High'].iloc[-1]))
        with col3:
            low_price = st.number_input("Low Price", min_value=0.0, value=float(df['Low'].iloc[-1]))
        
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            prediction = model.predict([[open_price, high_price, low_price]])
            st.success(f"**Predicted Closing Price:** {prediction[0]:.2f}")
            #st.write("Last actual close:", df['Close'].iloc[-1])

except FileNotFoundError:
    st.error("Error: NabilBank.csv file not found. Please make sure the file exists in the same directory.")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")