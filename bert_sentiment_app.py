import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="BERT Sentiment Analyzer", page_icon="💬", layout="wide")

st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🔍 BERT Sentiment Analysis App</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: grey;'>Analyze sentiment (Positive, Neutral, Negative) using BERT model</h4>", unsafe_allow_html=True)

# تحميل نموذج BERT للتحليل
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

nlp = load_model()

# واجهة المستخدم
text_input = st.text_area("📝 أدخل نصًا لتحليله", height=150)

if st.button("🔍 تحليل"):
    if text_input.strip() != "":
        with st.spinner("جارٍ التحليل..."):
            result = nlp(text_input)
            label = result[0]['label']
            score = result[0]['score']

            st.markdown(f"### ✅ النتيجة: **{label}**")
            st.markdown(f"### 🔢 النسبة: `{round(score*100, 2)}%`")

            if "1" in label or "2" in label:
                st.error("😞 سلبي")
            elif "3" in label:
                st.info("😐 محايد")
            else:
                st.success("😊 إيجابي")
    else:
        st.warning("من فضلك أدخل نصاً أولاً ❗")
