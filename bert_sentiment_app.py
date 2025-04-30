import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# تحميل النموذج والتوكن مع التخزين المؤقت
@st.cache_resource
def load_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # استخدام CPU فقط بوضع device=-1
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)
    return classifier

# تحليل المشاعر
def analyze_sentiment(text):
    classifier = load_model()
    result = classifier(text)[0]
    label_map = {
        'LABEL_0': 'سلبي 😠',
        'LABEL_1': 'محايد 😐',
        'LABEL_2': 'إيجابي 😊'
    }
    sentiment = label_map[result['label']]
    score = round(result['score'] * 100, 2)
    return sentiment, score

# واجهة Streamlit
st.set_page_config(page_title="تحليل المشاعر - RoBERTa", layout="centered")
st.title("🤖 RoBERTa Sentiment Analysis App")
st.markdown("**Analyse des sentiments (Positif, Neutre, Négatif) avec le modèle RoBERTa**")

text_input = st.text_area("📝 Entrez un texte à analyser", "")

if st.button("Analyser"):
    if text_input.strip():
        with st.spinner("Analyse en cours..."):
            sentiment, score = analyze_sentiment(text_input)
            st.success(f"**Résultat** : {sentiment} (confiance : {score}%)")
    else:
        st.warning("Veuillez entrer un texte.")
