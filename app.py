import streamlit as st
import joblib
import pandas as pd

st.title("🚗 Application de Prédiction - État du Véhicule")

# Chargement des fichiers joblib
@st.cache_resource
def load_assets():
    model = joblib.load("gb_model.joblib")
    scaler = joblib.load("scaler.joblib")
    encoders = joblib.load("encoders.joblib")
    uniques = joblib.load("uniques.joblib")
    return model, scaler, encoders, uniques

model, scaler, encoders, uniques = load_assets()

st.subheader("Entrez les données du véhicule :")

# Formulaire automatique selon le contenu de uniques.joblib
input_data = {}
for col, values in uniques.items():
    if isinstance(values, (list, tuple, pd.Series, set)):
        input_data[col] = st.selectbox(f"{col}", list(values))
    else:
        input_data[col] = st.number_input(f"{col}", value=0.0)

# Prédiction
if st.button("Prédire"):
    df = pd.DataFrame([input_data])
    
    # Application des encodeurs
    if isinstance(encoders, dict):
        for col, enc in encoders.items():
            if col in df.columns and hasattr(enc, "transform"):
                df[col] = enc.transform(df[col])
                
    # Normalisation et prédiction
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)
    
    st.success(f"Résultat : **{prediction[0]}**")
