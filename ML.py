# product_demand_forecasting.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st

# Load dataset
def load_data(path):
    data = pd.read_csv(path)
    return data

# Preprocess data
def preprocess_data(df):
    df = df.dropna()
    df = pd.get_dummies(df, drop_first=True)
    return df

# Train models
def train_models(X_train, y_train):
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, f"{name.replace(' ', '_').lower()}_model.pkl")
    return models

# Evaluate models
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, r2

# Plot predictions
def plot_predictions(y_test, predictions):
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=y_test, y=predictions)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted Demand")
    st.pyplot(plt)

# Streamlit UI
st.title("Product Demand Forecasting")

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write("Sample Data", df.head())

    df_processed = preprocess_data(df)

    if 'demand' not in df_processed.columns:
        st.error("Dataset must contain a 'demand' column as the target.")
    else:
        X = df_processed.drop("demand", axis=1)
        y = df_processed['demand']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = train_models(X_train, y_train)

        st.subheader("Model Evaluation")
        for name, model in models.items():
            mse, r2 = evaluate_model(model, X_test, y_test)
            st.write(f"**{name}** - MSE: {mse:.2f}, R2 Score: {r2:.2f}")
            st.write(f"Predictions by {name}:")
            plot_predictions(y_test, model.predict(X_test))

        st.success("Models trained and evaluated successfully.")
