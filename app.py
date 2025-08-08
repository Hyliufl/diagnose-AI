import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('Disease_symptom_and_patient_profile_dataset.csv')

# DATA PROCESSING AND PIPELINE
features = ['Fever', 'Cough', 'Difficulty Breathing', 'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level']
target = 'Disease'
X = df[features]
y = df[target]
categorical_cols = ['Fever', 'Cough', 'Difficulty Breathing', 'Gender', 'Blood Pressure', 'Cholesterol Level']
numeric_cols = ['Age']
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', StandardScaler(), numeric_cols)
])
pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)



# STREAMLIT
st.title("Health Input Recorder")

fields = {
    "Age": st.number_input("Age", min_value=0, max_value=120, step=1),
    "Cholesterol Level": st.selectbox("Cholesterol Level", ["Low", "Normal", "High"]),
    "Blood Pressure": st.selectbox("Blood Pressure", ["Low", "Normal", "High"]),
    "Gender": st.selectbox("Gender", ["Male", "Female"]),
    "Difficulty Breathing": st.radio("Difficulty Breathing", ["Yes", "No"]),
    "Fever": st.radio("Fever", ["Yes", "No"]),
    "Cough": st.radio("Cough", ["Yes", "No"])
}

description = st.text_area("description")

if st.button("Record Inputs"):
    user_inputs = fields
    user_df = pd.DataFrame([user_inputs])
    predicted_disease = pipeline.predict(user_df)[0]
    st.success(f"Based on your symptoms, the predicted disease is: **{predicted_disease}**")


#LLM
    import google.generativeai as genai
    genai.configure(api_key=st.secrets["AI_key"])
    model = genai.GenerativeModel("gemini-2.5-flash")
    data = pd.read_csv('medquad.csv', on_bad_lines='skip')

    prompt = f"""
    In a professional but also easy to understand voice, give a list of possible diseases or conditions that these symptoms
    indicate and with the help of the this data: {data}. Do not refer to "data" or "dataset".
    Here are the results from a classification model based on patient profiles and symptoms: {predicted_disease}
    List the items by starting with the one that is most relevant to the user and the most common. For each possibility, briefly explain why it may be relevant, how common it is
    and what you need to do if you have it.
    These are the symptoms: {user_inputs}. The patient also provided this description: {description}.
    """

    text1 = model.generate_content([prompt])
    st.write(text1.text)


