import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# --- Column mapping ---
COLUMN_NAME_MAP = {
    'Age': 'age',
    'Gender': 'gender',
    'Education Level': 'education_level',
    'Owns Agricultural Land': 'has_land',
    'Land in Acres': 'land_acres',
    "Parents' Occupation": 'parents_occupation',
    'Career Interest': 'interested_career',
    'Interested in Farming': 'is_interested_in_farming'
}

FEATURE_COLUMNS = [
    'age', 'gender', 'education_level', 'has_land',
    'land_acres', 'parents_occupation', 'interested_career'
]

CATEGORICAL_FEATURES = [
    'gender', 'education_level', 'has_land',
    'parents_occupation', 'interested_career'
]

# --- Train model function ---
@st.cache_resource
def train_model(filepath="response_500.csv"):
    df = pd.read_csv(filepath)
    df.rename(columns=COLUMN_NAME_MAP, inplace=True)

    df['land_acres'] = df['land_acres'].fillna(0)
    df['is_interested_in_farming'] = df['is_interested_in_farming'].apply(
        lambda x: 1 if str(x).strip() == 'Yes' else 0
    )

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 'Unknown'

    X = df[FEATURE_COLUMNS]
    y = df['is_interested_in_farming']

    preprocessor = ColumnTransformer(
        transformers=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)],
        remainder='passthrough'
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X, y)
    return pipeline

# --- Streamlit UI ---
st.set_page_config(page_title="Future of Farming Predictor", layout="centered")

st.title("🌱 Future of Farming Predictor")
st.write("Predicting youth's likelihood of choosing a farming career.")

# Train/load model
model_pipeline = train_model("response_500.csv")

# Input form
st.subheader("Enter a hypothetical profile:")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=15, max_value=40, value=22)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    education_level = st.selectbox("Education Level", ["School", "Intermediate", "Undergraduate", "Postgraduate", "PhD"])

with col2:
    parents_occupation = st.selectbox("Parents' Occupation", ["Farming", "Daily Wage Labour", "Government Job", "Private Job", "Business"])
    interested_career = st.selectbox("Interested Career", ["Farmer", "Software Engineer", "Teacher", "Doctor", "Government Job", "Scientist", "Entrepreneur"])
    has_land = st.selectbox("Access to Land?", ["Yes", "No"])

land_acres = st.number_input("Land Owned (acres)", min_value=0.0, value=5.0, step=0.1)

# Predict button
if st.button("Predict Career Choice"):
    profile = {
        "age": age,
        "gender": gender,
        "education_level": education_level,
        "parents_occupation": parents_occupation,
        "interested_career": interested_career,
        "has_land": has_land,
        "land_acres": land_acres
    }

    input_df = pd.DataFrame([profile])[FEATURE_COLUMNS]
    prediction_proba = model_pipeline.predict_proba(input_df)[0]
    prediction_class = int(model_pipeline.predict(input_df)[0])

    prediction_text = "🌾 Choosing a Farming Career" if prediction_class == 1 else "❌ Not Choosing a Farming Career"
    confidence = prediction_proba[prediction_class]

    st.success(f"**Prediction:** {prediction_text}")
    st.info(f"**Confidence:** {confidence:.2%}")
