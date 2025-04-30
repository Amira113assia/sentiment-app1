import streamlit as st
from transformers import pipeline

# إعداد الصفحة
st.set_page_config(page_title="RoBERTa Sentiment Analyzer", page_icon="🤖", layout="wide")

# عنوان التطبيق
st.markdown("<h1 style='text-align: center; color: #6A5ACD;'>🤖 RoBERTa Sentiment Analysis App</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: grey;'>Analyse des sentiments (Positif, Neutre, Négatif) avec le modèle RoBERTa</h4>", unsafe_allow_html=True)

# تحميل النموذج
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

model = load_model()

# حقل الإدخال
text_input = st.text_area("📝 Entrez un texte à analyser", height=150)

# دالة التحليل
def analyze_sentiment(text):
    result = model(text)[0]
    label = result['label']
    score = result['score']
    
    # تصنيف المشاعر حسب النموذج
    if label == "LABEL_0":
        sentiment = "😞 Négatif"
    elif label == "LABEL_1":
        sentiment = "😐 Neutre"
    else:
        sentiment = "😊 Positif"

    return sentiment, score

# زر التحليل
if st.button("🔍 Analyser"):
    if text_input.strip():
        with st.spinner("Analyse en cours..."):
            sentiment, score = analyze_sentiment(text_input)
            st.markdown(f"### ✅ Résultat : **{sentiment}**")
            st.markdown(f"### 🔢 Score de confiance : `{score:.2%}`")
    else:
        st.warning("⚠️ Veuillez entrer un texte avant d'analyser.")
