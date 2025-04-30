import streamlit as st
from transformers import pipeline

# --- إعداد الواجهة ---
st.set_page_config(page_title="RoBERTa Sentiment App", page_icon="🤖", layout="centered")

# --- تحميل النموذج (مع تخزين في الكاش) ---
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

model = load_model()

# --- عنوان التطبيق ---
st.title("🤖 RoBERTa Sentiment Analysis App")
st.markdown("### Analyse des sentiments (Positif ou Négatif) avec le modèle DistilBERT")

# --- إدخال النص ---
text_input = st.text_area("📝 Entrez un texte à analyser", height=150)

# --- زر التحليل ---
if st.button("Analyser le texte"):
    if not text_input.strip():
        st.warning("⚠️ Veuillez entrer un texte à analyser.")
    else:
        try:
            result = model(text_input)[0]
            label = result['label']
            score = round(result['score'] * 100, 2)

            if label == "POSITIVE":
                st.success(f"😊 Sentiment détecté : **Positif** ({score}%)")
            elif label == "NEGATIVE":
                st.error(f"😞 Sentiment détecté : **Négatif** ({score}%)")
            else:
                st.info(f"😐 Sentiment détecté : **{label}** ({score}%)")

        except Exception as e:
            st.error("❌ Une erreur est survenue lors de l'analyse.")
            st.exception(e)

# --- Footer ---
st.markdown("---")
st.markdown("Développé par **Amira ❤️** - Powered by Hugging Face & Streamlit")
