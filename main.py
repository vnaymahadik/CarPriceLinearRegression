import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Set page configuration
st.set_page_config(page_title="Vehicle Price Prediction App", layout="wide")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'le_dict' not in st.session_state:
    st.session_state.le_dict = {}

# Function to load and preprocess data
def load_and_preprocess_data(file):
    data = pd.read_csv(file)
    data = data.dropna()  # Remove rows with NaN values
    return data

# Function to train the model
def train_model(data, test_size):
    features = ['Name', 'Location', 'Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission']
    target = 'Price'

    # Encode categorical variables
    for col in ['Name', 'Location', 'Fuel_Type', 'Transmission']:
        le = LabelEncoder()
        data[col + '_encoded'] = le.fit_transform(data[col])
        st.session_state.le_dict[col] = le

    X = data[[col + '_encoded' if col in ['Name', 'Location', 'Fuel_Type', 'Transmission'] else col for col in features]]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2

# Main app
def main():
    st.title("Vehicle Price Prediction App")

    tabs = st.tabs(["How to Use", "Exploratory Data Analysis", "Model Evaluation", "Prediction"])

    with tabs[0]:
        st.header("How to Use This App")
        st.write("""
        Welcome to the Vehicle Price Prediction App! This application uses Linear Regression to predict Vehicle prices based on various features.

        Here's how to use the app:
        1. Upload a CSV file containing your dataset.
        2. Explore the data in the 'Exploratory Data Analysis' tab.
        3. Train and evaluate the model in the 'Model Evaluation' tab.
        4. Make predictions in the 'Prediction' tab.

        The dataset should include various features of Vehicle and their price.
        """)

    with tabs[1]:
        st.header("Exploratory Data Analysis")
        file = st.file_uploader("Upload your CSV file", type="csv")
        if file is not None:
            st.session_state.data = load_and_preprocess_data(file)
            st.write("Sample Data:")
            st.write(st.session_state.data.head())

            st.write("Data Summary:")
            st.write(st.session_state.data.describe())

            st.write("Histograms:")
            for col in ['Year', 'Kilometers_Driven', 'Price']:
                fig = px.histogram(st.session_state.data, x=col, title=f"Histogram of {col}")
                st.plotly_chart(fig)

            st.write("Correlation Matrix:")
            corr_matrix = st.session_state.data[['Year', 'Kilometers_Driven', 'Price']].corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto")
            st.plotly_chart(fig)

    with tabs[2]:
        st.header("Model Evaluation")
        if st.session_state.data is not None:
            test_size = st.slider("Select test data percentage", 10, 50, 20) / 100
            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    st.session_state.model, mse, r2 = train_model(st.session_state.data, test_size)
                st.success("Model trained successfully!")
                st.write(f"Mean Squared Error: {mse:.2f}")
                st.write(f"R-squared Score: {r2:.2f}")
        else:
            st.warning("Please upload a CSV file in the 'Exploratory Data Analysis' tab.")

    with tabs[3]:
        st.header("Prediction")
        if st.session_state.model is not None and st.session_state.data is not None:
            st.write("Enter vehicle details to predict price:")
            name = st.selectbox("Name", options=st.session_state.data['Name'].unique())
            location = st.selectbox("Location", options=st.session_state.data['Location'].unique())
            year = st.number_input("Year", min_value=1900, max_value=2023, value=2020)
            km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)
            fuel_type = st.selectbox("Fuel Type", options=st.session_state.data['Fuel_Type'].unique())
            transmission = st.selectbox("Transmission", options=st.session_state.data['Transmission'].unique())

            if st.button("Predict Car Price"):
                input_data = pd.DataFrame({
                    'Name_encoded': [st.session_state.le_dict['Name'].transform([name])[0]],
                    'Location_encoded': [st.session_state.le_dict['Location'].transform([location])[0]],
                    'Year': [year],
                    'Kilometers_Driven': [km_driven],
                    'Fuel_Type_encoded': [st.session_state.le_dict['Fuel_Type'].transform([fuel_type])[0]],
                    'Transmission_encoded': [st.session_state.le_dict['Transmission'].transform([transmission])[0]]
                })
                prediction = st.session_state.model.predict(input_data)[0]
                st.success(f"Predicted Car Price: ₹{prediction:.2f} Lakh")
        else:
            st.warning("Please train the model in the 'Model Evaluation' tab.")

if __name__ == "__main__":
    main()