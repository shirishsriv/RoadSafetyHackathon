# ---------------------------------------------------------------------
# 🚦 ROAD SAFETY GPT ADVISOR (v4.9) — DEBUGGED & IMPROVED
# ---------------------------------------------------------------------

import os
import base64
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import google.generativeai as genai


# ---------------------------------------------------------------------
# ✅ STREAMLIT CONFIG
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Road Safety Advisor — DPDP Consent",
    page_icon="🚧",
    layout="wide"
)


# ---------------------------------------------------------------------
# ✅ DPDP CONSENT OVERLAY (Fixed version)
# ---------------------------------------------------------------------
if "consent_given" not in st.session_state:
    st.session_state.consent_given = False

if not st.session_state.consent_given:
    st.markdown("""
    <style>
        .overlay {
            position: fixed; top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(0, 0, 0, 0.6);
            z-index: 9999;
        }
        .popup {
            position: fixed; top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            width: 420px;
            padding: 28px;
            border-radius: 12px;
            text-align: center;
            z-index: 10000;
        }
    </style>

    <div class="overlay">
        <div class="popup">
    """, unsafe_allow_html=True)

    st.write("### 🔒 Data Consent (DPDP Act — India)")
    st.write("""
    This app sends your input to **Google Gemini (3rd-party AI)**  
    for generating road-safety recommendations.

    ✅ No data stored  
    ⚠ Do NOT enter personal or sensitive information  
    """)

    consent = st.checkbox("I give consent to share my input with Gemini")

    if st.button("Proceed"):
        if consent:
            st.session_state.consent_given = True
            st.rerun()
        else:
            st.error("✅ Please check consent to continue.")

    st.markdown("</div></div>", unsafe_allow_html=True)
    st.stop()


# ---------------------------------------------------------------------
# 🎨 HEADER IMAGE (with fallback)
# ---------------------------------------------------------------------
def get_base64_image(path):
    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode()


try:
    image = get_base64_image("cars.png")
    st.markdown(f"""
    <style>
        .header {{
            background-image: url("data:image/png;base64,{image}");
            background-size: cover;
            height: 130px;
            border-radius: 12px;
            position: relative;
        }}
        .header-overlay {{
            position: absolute; inset: 0;
            background: rgba(255, 255, 255, 0.75);
            border-radius: 12px;
        }}
        .header-text {{
            position: absolute;
            width: 100%;
            text-align: center;
            top: 35%;
            font-size: 1.8rem;
            font-weight: 700;
            color: #FF8F00;
        }}
    </style>

    <div class="header">
      <div class="header-overlay"></div>
      <div class="header-text">🚦 Road Safety Advisor</div>
    </div>
    """, unsafe_allow_html=True)
except FileNotFoundError:
    st.title("🚦 Road Safety Advisor")


# ---------------------------------------------------------------------
# 🔐 GEMINI API CONFIG
# ---------------------------------------------------------------------
st.sidebar.header("🔐 Gemini AI Settings")
gemini_key = st.sidebar.text_input("Enter Gemini API Key:", type="password")

gemini_model = None
if gemini_key:
    genai.configure(api_key=gemini_key)
    gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")
    st.sidebar.success("✅ Gemini Ready")
else:
    st.sidebar.warning("⚠ Gemini AI disabled until key is added")


# ---------------------------------------------------------------------
# 📤 LOAD CSV (with caching)
# ---------------------------------------------------------------------
@st.cache_data
def load_interventions(file=None):
    df = pd.read_csv(file) if file else pd.read_csv("interventions.csv")
    df.columns = df.columns.str.replace(" ", "_").str.lower()
    df["searchable_text"] = df.astype(str).agg(" ".join, axis=1)
    return df


uploaded = st.sidebar.file_uploader("Upload interventions.csv", type=["csv"])
df = load_interventions(uploaded) if uploaded else load_interventions()


# ---------------------------------------------------------------------
# 🔍 VECTOR SEARCH ENGINE (with caching)
# ---------------------------------------------------------------------
@st.cache_resource
def build_tfidf_model(df):
    vect = TfidfVectorizer(max_features=6000, stop_words="english")
    tfidf = vect.fit_transform(df["searchable_text"])
    model = NearestNeighbors(metric="cosine").fit(tfidf)
    return vect, model, tfidf


vectorizer, nbrs, tfidf_matrix = build_tfidf_model(df)


# ---------------------------------------------------------------------
# 🧾 PROBLEM INPUT FORM (Prevents reruns while typing)
# ---------------------------------------------------------------------
with st.form("query_form"):
    st.header("📝 Input Road Safety Problem Details")

    safety_issues = st.text_area("⚠ Safety Issue *", height=80)
    environment = st.text_area("🌍 Environment / Road Conditions *", height=70)
    detail = st.text_area("🔍 Detailed Description *", height=80)

    submitted = st.form_submit_button("🔎 Identify Suitable Interventions")


# ---------------------------------------------------------------------
# ✅ PROCESS QUERY
# ---------------------------------------------------------------------
if submitted:
    if not (safety_issues.strip() and environment.strip() and detail.strip()):
        st.error("⚠ Please fill in all required fields.")
        st.stop()

    query = f"{safety_issues} {environment} {detail}"
    distances, indices = nbrs.kneighbors(vectorizer.transform([query]), n_neighbors=3)

    st.subheader("✅ Top Recommended Interventions")

    for idx, row_idx in enumerate(indices[0]):
        row = df.iloc[row_idx]

        st.markdown("---")
        st.write(f"### #{idx+1} — {row['problem_category']} ({row['type']})")
        st.write(row["data"])

        if gemini_model:
            with st.expander("🤖 AI Explanation"):
                try:
                    response = gemini_model.generate_content(
                        f"Problem: {detail}\nIntervention: {row['data']}\nExplain benefit in 5 sentences."
                    )
                    text = response.text if hasattr(response, "text") else str(response)
                    st.write(text)
                except Exception as e:
                    st.error(f"⚠ AI Error: {e}")

st.caption("🛡 DPDP Consent | AI Reasoning | CSV Data Search | v4.9")
