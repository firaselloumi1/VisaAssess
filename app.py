import os
import re
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="VisasAsses",
    page_icon="🛂",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Light Modern Theme CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'DM Sans', sans-serif !important;
    }
    
    .stApp {
        background: #f8f9fa !important;
    }
    
    .main-container {
        max-width: 420px;
        margin: 0 auto;
        padding: 1.5rem;
    }
    
    .logo-icon {
        text-align: center;
        font-size: 3.5rem;
        margin-bottom: 0.5rem;
    }
    
    .app-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        letter-spacing: -0.5px;
        margin-bottom: 0.3rem;
    }
    
    .app-subtitle {
        text-align: center;
        color: #64748b;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    
    .input-card {
        background: white;
        border-radius: 24px;
        padding: 1.5rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.06);
    }
    
    .drop-zone {
        border: 2px dashed #cbd5e1;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
        margin-bottom: 0.8rem;
    }
    
    .drop-zone:hover {
        border-color: #6366f1;
        background: #f8fafc;
    }
    
    .drop-icon {
        font-size: 2.5rem;
    }
    
    .drop-text {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }
    
    .or-text {
        text-align: center;
        color: #cbd5e1;
        font-size: 0.75rem;
        margin: 0.8rem 0;
    }
    
    .stTextArea textarea {
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 14px !important;
        color: #1e293b !important;
        font-size: 0.9rem !important;
        padding: 1rem !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #94a3b8 !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
    }
    
    .analyze-btn {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 1rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        margin-top: 0.5rem !important;
    }
    
    .analyze-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.3) !important;
    }
    
    .result-card {
        background: white;
        border-radius: 24px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
        animation: slideUp 0.4s ease-out;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .result-emoji {
        font-size: 4.5rem;
        margin-bottom: 0.3rem;
    }
    
    .result-text {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: 1px;
    }
    
    .result-accepted {
        color: #10b981;
    }
    
    .result-refused {
        color: #ef4444;
    }
    
    .confidence-badge {
        display: inline-block;
        background: #f1f5f9;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 0.8rem;
    }
    
    .confidence-value {
        color: #1e293b;
        font-weight: 600;
    }
    
    .features-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        justify-content: center;
        margin-top: 1.2rem;
    }
    
    .feature-chip {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        border-radius: 20px;
        padding: 0.4rem 0.8rem;
        font-size: 0.75rem;
        color: #475569;
        font-weight: 500;
    }
    
    .how-to-card {
        background: white;
        border-radius: 16px;
        padding: 1.2rem;
        margin-top: 1rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    }
    
    .how-to-title {
        color: #1e293b;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
    
    .how-to-item {
        color: #64748b;
        font-size: 0.8rem;
        padding: 0.25rem 0;
    }
    
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.7rem;
        margin-top: 1.5rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    DATA_DIR = "visa_data"
    
    texts, labels = [], []
    for label in ["accepted", "refused"]:
        folder = os.path.join(DATA_DIR, label)
        for f in os.listdir(folder):
            if f.endswith(".txt"):
                with open(os.path.join(folder, f), 'r', encoding='utf-8', errors='ignore') as file:
                    text = file.read().lower().replace('é','e').replace('è','e').replace('à','a')
                    texts.append(text)
                labels.append(1 if label == "refused" else 0)
    
    labels = np.array(labels)
    
    def extract_features(text):
        m = re.findall(r'compte.bloque.*?(\d+)', text)
        funds = int(m[0]) if m else 0
        return [1 if funds >= 20000 else 0, 1 if funds < 15000 else 0,
                1 if any(w in text for w in ['hotel', 'reservation']) else 0,
                1 if any(w in text for w in ['foyer', 'bail', 'studio']) else 0,
                1 if 'accord' in text and 'prealable' in text else 0,
                1 if 'attestation' in text else 0, 1 if 'bac' in text else 0]
    
    vectorizer = TfidfVectorizer(max_features=300, ngram_range=(1, 2), min_df=1, max_df=0.9)
    X_tfidf = vectorizer.fit_transform(texts)
    extra_features = np.array([extract_features(t) for t in texts])
    X_combined = np.hstack([X_tfidf.toarray(), extra_features])
    
    model = VotingClassifier(estimators=[
        ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.5)),
        ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=3)),
        ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', C=0.5))
    ], voting='soft')
    
    model.fit(X_combined, labels)
    return model, vectorizer, extract_features

model, vectorizer, extract_features = load_model()

# Main container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header
st.markdown('''
    <div class="logo-icon">🛂</div>
    <div class="app-title">VisasAsses</div>
    <div class="app-subtitle">AI-Powered Visa Prediction</div>
''', unsafe_allow_html=True)

# Input card
st.markdown('<div class="input-card">', unsafe_allow_html=True)

# Upload
st.markdown('''
    <div class="drop-zone">
        <div class="drop-icon">📄</div>
        <div class="drop-text">Drop .txt files here</div>
    </div>
''', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "",
    type=['txt'],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

st.markdown('<div class="or-text">— OR —</div>', unsafe_allow_html=True)

# Text input
text_input = st.text_area(
    "",
    placeholder="Accord prealable Campus France\nCompte bloque 25000 euros\nContrat de bail...",
    label_visibility="collapsed"
)

# Get user text
if uploaded_files:
    combined_text = ""
    for file in uploaded_files:
        content = file.getvalue().decode("utf-8", errors="ignore")
        combined_text += content + "\n"
    user_text = combined_text
elif text_input:
    user_text = text_input
else:
    user_text = None

# Predict button
if st.button("✨ Analyze My Visa"):
    if user_text:
        try:
            text_norm = user_text.lower().replace('é','e').replace('è','e').replace('à','a')
            X_new_tfidf = vectorizer.transform([text_norm])
            X_new = np.hstack([X_new_tfidf.toarray(), np.array([extract_features(text_norm)])])
            
            pred = model.predict(X_new)[0]
            prob = model.predict_proba(X_new)[0]
            confidence = prob[pred] * 100
            
            if pred == 0:
                emoji = "🎉"
                text = "ACCEPTED"
                cls = "result-accepted"
            else:
                emoji = "😔"
                text = "REFUSED"
                cls = "result-refused"
            
            # Features
            features = []
            if 'compte bloque' in text_norm:
                m = re.findall(r'compte.bloque.*?(\d+)', text_norm)
                if m:
                    features.append(f"💰 {int(m[0]):,}€")
            if any(w in text_norm for w in ['foyer', 'bail', 'studio']):
                features.append("🏠 Housing")
            if any(w in text_norm for w in ['hotel', 'reservation']):
                features.append("🏨 Hotel")
            if 'accord' in text_norm and 'prealable' in text_norm:
                features.append("📜 Campus")
            if 'bac' in text_norm:
                features.append("🎓 Education")
            
            features_html = ""
            for f in features:
                features_html += f'<span class="feature-chip">{f}</span>'
            
            st.markdown(f'''
                <div class="result-card">
                    <div class="result-emoji">{emoji}</div>
                    <div class="result-text {cls}">{text}</div>
                    <div class="confidence-badge">Confidence: <span class="confidence-value">{confidence:.0f}%</span></div>
                    <div class="features-row">{features_html}</div>
                </div>
            ''', unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter your application details")

st.markdown('</div>', unsafe_allow_html=True)

# How to use - styled as a card instead of expander
st.markdown('''
    <div class="how-to-card">
        <div class="how-to-title">📋 How to use</div>
        <div class="how-to-item">💰 Include: Financial (blocked account amount)</div>
        <div class="how-to-item">🏠 Housing: Hotel, foyer, bail, studio</div>
        <div class="how-to-item">📜 Documents: Accord préalable, attestation</div>
        <div class="how-to-item">🎓 Education: Baccalauréat, relevés</div>
    </div>
''', unsafe_allow_html=True)

st.markdown('<div class="footer">Trained on 27 cases • ~74% accuracy</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
