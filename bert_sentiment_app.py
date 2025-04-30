import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TextClassificationPipeline
import torch
import numpy as np

# إعداد الصفحة
st.set_page_config(page_title="RoBERTa Sentiment", page_icon="🤖", layout="wide")
st.markdown("<h1 style='text-align: center; color: #6A5ACD;'>🔍 Application d'Analyse des Sentiments RoBERTa 🔍</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: grey;'>Analyse des sentiments (Positif, Neutre, Négatif) avec le modèle RoBERTa</h4>", unsafe_allow_html=True)

# تحميل النموذج و Tokenizer
@st.cache_resource
def load_roberta():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_roberta()

# إنشاء pipeline يدوي
def analyze_sentiment(text):
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True)
    with torch.no_grad():
        output = model(**encoded_input)
    scores = torch.nn.functional.softmax(output.logits, dim=1).numpy()[0]
    labels = ['Négatif 😞', 'Neutre 😐', 'Positif 😀']
    max_idx = np.argmax(scores)
    return labels[max_idx], scores[max_idx]

# واجهة المستخدم
text = st.text_area("📝 Entrez un texte à analyser", height=150)

if st.button("🔍 Analyser"):
    if text.strip() != "":
        with st.spinner("Analyse en cours..."):
            label, score = analyze_sentiment(text)
            st.markdown(f"### ✅ Résultat : **{label}**")
            st.markdown(f"### 🔢 Score : `{score*100:.2f}%`")
    else:
        st.warning("❗ Veuillez entrer un texte d'abord.")
