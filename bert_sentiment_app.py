import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="BERT Sentiment Analyzer", page_icon="💬", layout="wide")

st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🔍 Application d'Analyse des Sentiments BERT 🔍</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: grey;'>Analyse des sentiments (Positif, Neutre, Négatif) avec le modèle BERT</h4>", unsafe_allow_html=True)

# Chargement du modèle BERT pour l'analyse
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

nlp = load_model()

# Interface utilisateur
text_input = st.text_area("📝 Entrez un texte à analyser", height=150)

if st.button("🔍 Analyser"):
    if text_input.strip() != "":
        with st.spinner("Analyse en cours..."):
            result = nlp(text_input)
            label = result[0]['label']
            score = result[0]['score']

            st.markdown(f"### ✅ Résultat : **{label}**")
            st.markdown(f"### 🔢 Score : `{round(score*100, 2)}%`")

            if "1" in label or "2" in label:
                st.error("😞 Négatif")
            elif "3" in label:
                st.info("😐 Neutre")
            else:
                st.success("😊 Positif")
    else:
        st.warning("Veuillez entrer un texte d'abord ❗")
