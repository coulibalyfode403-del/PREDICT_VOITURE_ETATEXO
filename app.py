import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.title("🚗 Application de Prédiction - État du Véhicule")

# Chargement des fichiers joblib avec mise en cache
@st.cache_resource
def load_assets():
    model = joblib.load("gb_model.joblib")
    scaler = joblib.load("scaler.joblib")
    encoders = joblib.load("encoders.joblib")
    uniques = joblib.load("uniques.joblib")
    return model, scaler, encoders, uniques

model, scaler, encoders, uniques = load_assets()

st.subheader("Entrez les données du véhicule :")

# Ordre exact des variables catégorielles sauvegardées dans uniques.joblib
# Dans votre notebook : uniques = [Marque, Transmission, Quartier, Etat] (cat_data)
# On exclut la variable cible 'Etat' (index 3) pour l'entrée utilisateur
cat_cols = ['Marque', 'Transmission', 'Quartier']

input_data = {}

# 1. Sélection pour les variables catégorielles (depuis uniques.joblib)
for idx, col in enumerate(cat_cols):
    input_data[col] = st.selectbox(f"Sélectionnez {col}", uniques[idx])

# 2. Saisie pour les variables numériques
input_data['Année'] = st.number_input("Année du véhicule", min_value=1990, max_value=2026, value=2015)
input_data['Prix'] = st.number_input("Prix du véhicule (FCFA)", min_value=0, value=5000000, step=100000)

# Prédiction
if st.button("Prédire"):
    # Création du DataFrame dans l'ordre exact d'entraînement : ['Marque', 'Année', 'Transmission', 'Prix', 'Quartier']
    df_input = pd.DataFrame([{
        'Marque': input_data['Marque'],
        'Année': input_data['Année'],
        'Transmission': input_data['Transmission'],
        'Prix': input_data['Prix'],
        'Quartier': input_data['Quartier']
    }])

    # 1. Transformation des variables catégorielles avec les LabelEncoders
    # Dans votre notebook : encoders = [encoder_Marque, encoder_Transmission, encoder_Quartier, encoder_Etat]
    for idx, col in enumerate(cat_cols):
        df_input[col] = encoders[idx].transform(df_input[col])

    # 2. Normalisation avec le Scaler
    df_scaled = scaler.transform(df_input)

    # 3. Prédiction
    prediction = model.predict(df_scaled)
    
    # Décoder la prédiction (0 = D'occasion, 1 = Venant)
    label_etat = encoders[3].inverse_transform(prediction)[0]

    st.success(f"Résultat de la prédiction : **{label_etat}**")