# ✅ الكود الأصلي بدون تصميم احترافي

import streamlit as st
from transformers import pipeline

# تحميل نموذج RoBERTa
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

st.title("تحليل المشاعر باستخدام RoBERTa")

text = st.text_input("أدخل جملة لتحليل المشاعر:")

if st.button("تحليل"):
    if text:
        result = sentiment_pipeline(text)[0]
        st.write(f"النتيجة: {result['label']} (بثقة {round(result['score'], 2)})")
    else:
        st.warning("يرجى إدخال جملة أولاً")


# ✅ الكود بعد التعديل بواجهة احترافية

import streamlit as st
from transformers import pipeline

# تحميل نموذج RoBERTa
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return result['label'], result['score']

# 💡 تصميم احترافي
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>💬 RoBERTa Sentiment Analyzer</h1>
    <p style='text-align: center;'>أدخل جملة باللغة الإنجليزية وسنحللها لك</p>
""", unsafe_allow_html=True)

user_input = st.text_area("👇 اكتب جملتك هنا:", height=150)

if st.button("🔍 تحليل المشاعر"):
    if user_input.strip():
        label, score = analyze_sentiment(user_input)

        if label == "LABEL_2":
            st.success(f"💚 النتيجة: إيجابي (ثقة {round(score, 2)})")
        elif label == "LABEL_0":
            st.error(f"💔 النتيجة: سلبي (ثقة {round(score, 2)})")
        else:
            st.info(f"😐 النتيجة: محايد (ثقة {round(score, 2)})")
    else:
        st.warning("⚠️ من فضلك أدخل جملة أولاً.")

