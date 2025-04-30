iimport streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# تحميل النموذج والمُرمِّز الخاص بـ RoBERTa
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# إعداد واجهة المستخدم Streamlit
st.set_page_config(page_title="RoBERTa Sentiment Analysis App", page_icon="💬", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🔍 Application d'Analyse des Sentiments RoBERTa 🔍</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: grey;'>Analyse des sentiments (Positif, Neutre, Négatif) avec le modèle RoBERTa</h4>", unsafe_allow_html=True)

# دالة لتحليل المشاعر
def analyze_sentiment(text):
    # تحويل النص إلى مدخلات للنموذج
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True)
    
    # التأكد أن النموذج لا يتعامل مع GPU فقط إذا كان متاحًا
    with torch.no_grad():
        output = model(**encoded_input)
        
    # التأكد من أن الـ logits على الـ CPU قبل التحويل
    logits = output.logits.detach().cpu()
    
    # حساب الاحتمالات باستخدام softmax
    scores = torch.nn.functional.softmax(logits, dim=1)[0].numpy()
    
    # التصنيف والتسميات
    labels = ['Négatif 😞', 'Neutre 😐', 'Positif 😀']
    max_idx = np.argmax(scores)
    
    return labels[max_idx], scores[max_idx]

# واجهة المستخدم لإدخال النص
text_input = st.text_area("📝 Entrez un texte à analyser", height=150)

# تنفيذ التحليل عند الضغط على الزر
if st.button("🔍 Analyse"):
    if text_input.strip() != "":
        with st.spinner("Analyse en cours..."):
            label, score = analyze_sentiment(text_input)
            st.markdown(f"### ✅ Résultat : **{label}**")
            st.markdown(f"### 🔢 Score : `{round(score * 100, 2)}%`")
            
            if label == "Négatif 😞":
                st.error("😞 Négatif")
            elif label == "Neutre 😐":
                st.info("😐 Neutre")
            else:
                st.success("😊 Positif")
    else:
        st.warning("❗ Veuillez entrer un texte avant d'analyser.")
