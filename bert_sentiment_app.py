import streamlit as st
from transformers import pipeline

# تحميل نموذج BERT لتحليل المشاعر
sentiment_pipeline = pipeline("sentiment-analysis")

# عنوان التطبيق
st.title("تحليل المشاعر باستخدام BERT")
st.write("📍 هذا التطبيق يحلل ما إذا كان النص سلبيًا أو إيجابيًا أو محايدًا.")

# إدخال المستخدم
user_input = st.text_area("أدخل نصًا هنا:")

# زر التحليل
if st.button("🔍 تحليل"):
    if user_input.strip() != "":
        result = sentiment_pipeline(user_input)[0]
        label = result['label']
        score = result['score']

        # عرض النتيجة
        st.subheader("📊 النتيجة:")
        st.write(f"**التصنيف**: {label}")
        st.write(f"**درجة الثقة**: {score:.2f}")
    else:
        st.warning("⚠️ الرجاء إدخال نص قبل التحليل.")
