import streamlit as st
from transformers import pipeline

# إعداد الواجهة
st.set_page_config(page_title="RoBERTa Sentiment Analyzer", page_icon="🤖", layout="wide")

st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🤖 RoBERTa Sentiment Analysis App</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: grey;'>Analyse des sentiments (Positif, Neutre, Négatif) avec le modèle RoBERTa</h4>", unsafe_allow_html=True)

# تحميل نموذج RoBERTa المدرب على تويتر
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

nlp = load_model()

# إدخال المستخدم
text_input = st.text_area("📝 Entrez un texte à analyser", height=150)

if st.button("🔍 Analyser"):
    if text_input.strip() != "":
        with st.spinner("Analyse en cours..."):
            result = nlp(text_input)[0]
            label = result['label']
            score = result['score']

            # الترجمة حسب RoBERTa
            if label == "LABEL_0":
                sentiment = "😞 Négatif"
            elif label == "LABEL_1":
                sentiment = "😐 Neutre"
            else:
                sentiment = "😊 Positif"

            st.markdown(f"### ✅ Résultat: {sentiment}")
            st.markdown(f"### 📊 Score: `{score:.2%}`")
    else:
        st.warning("Veuillez entrer un texte d'abord ❗")
