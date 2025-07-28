import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
# Titre
st.title("Prédiction Inclusion financiere en Afrique (Banque)")
st.header("Classification Machine learning")
st.subheader("Avec streamlit" )
st.text("Bienvenue dans le portail de classification!!!")

# Charger modèle de prediction
model = joblib.load("model.pkl")
feature_names = joblib.load("features.pkl")

# Champs de saisie desninfo
region = st.selectbox("Région", ['Rwanda', 'Tanzania', 'Uganda'])
cellphone_access = st.number_input("acces tel", 0.0)
household_size = st.number_input("Taille ménage", 0.0)
age_of_respondent = st.number_input("Age", 0.0)
marital_status = st.selectbox("Statut marital", ['marital_status_Dont know', 'marital_status_Married/Living together',
                                                 'marital_status_Single/Never Married', 'marital_status_Widowed'])
gender_of_respondent = st.number_input("Sexe", 0.0)
location_type = st.number_input("type de d'hanitation", 0.0)
relationship_with_head = st.selectbox("position dans le ménage", ['relationship_with_head_Head of Household',
                                                                  'relationship_with_head_Other non-relatives',
                                                                  'education_level_Vocational/Specialised training'])
educational_level = st.selectbox("Niveau d'éducation", ['education_level_Dont know', 'education_level_Primary education',
                                                     'education_level_Secondary education',
                                                     'education_level_Tertiary education',
                                                     'education_level_Vocational/Specialised training'])
job_type = st.selectbox("Emploi", ['job_type_Farming and Fishing', 'job_type_Formally employed Government',
                                     'job_type_Formally employed Private', 'job_type_Government Dependent',
                                     'job_type_Informally employed', 'job_type_No Income', 'job_type_Other Income',
                                     'job_type_Remittance Dependent', 'job_type_Self employed'])

# Construction de l'input
input_dict = {
    'region ': region ,
    'cellphone_access': cellphone_access,
    'household_size': household_size,
    'age_of_respondent': age_of_respondent,
    'marital_status ': marital_status ,
    'gender_of_respondent': gender_of_respondent,
    'location_type': location_type,
    'relationship_with_head': relationship_with_head,
    'educational_level': educational_level,
    'job_type': job_type
}

# Encodage one-hot manuel
for col in feature_names:
    if col not in input_dict:
        input_dict[col] = 0

# Remplir les colonnes one-hot selon la sélection
input_dict[f'region_{region}'] = 1
input_dict[f'marital_status_{marital_status }'] = 1
input_dict[f'educational_level_{educational_level}'] = 1
input_dict[f'relationship_with_head_{relationship_with_head}'] = 1
input_dict[f'job_type_{job_type}'] = 1

# Transformer en DataFrame
X_input = pd.DataFrame([input_dict])[feature_names]

# Prédire
if st.button("Prédire banque"):
    prediction = model.predict(X_input)[0]
    if prediction == 1:
        st.error("✅La personne est susceptible de détenir un compte bancaire.")
    else:
        st.success(" ⚠️ La personne n'est pas susceptible de détenir un compte bancaire.")






