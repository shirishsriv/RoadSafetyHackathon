# ---------------------------------------------------------------------
# 🚦 ROAD SAFETY GPT ADVISOR VERSION — DPDP CONSENT POPUP INTEGRATED
# ---------------------------------------------------------------------

import os
import sys
import io
import base64
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import google.generativeai as genai
import folium
from streamlit_folium import st_folium

# ---------------------------------------------------------------------
# ✅ STREAMLIT CONFIG
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Road Safety Advisor v4.8 — Hackathon Edition",
    page_icon="🚧",
    layout="wide"
)

# ---------------------------------------------------------------------
# ✅ SESSION STATE FOR CONSENT
# ---------------------------------------------------------------------
if "consent_given" not in st.session_state:
    st.session_state.consent_given = False


# --------------------------------------------------------
# ✅ CONSENT STATE MANAGEMENT
# --------------------------------------------------------
if "consent_given" not in st.session_state:
    st.session_state.consent_given = False

# --------------------------------------------------------
# ✅ CONSENT POPUP (Compatible with all Streamlit versions)
# --------------------------------------------------------
def consent_popup():
    with st.container():
        st.markdown(
            """
            <div style="
                position: fixed;
                top: 0; left: 0;
                width: 100%; height: 100%;
                background-color: rgba(0,0,0,0.55);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 9999;
            ">
                <div style="
                    background: white;
                    padding: 30px;
                    width: 450px;
                    border-radius: 12px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
                    text-align: center;
                    font-family: Arial, sans-serif;
                ">
                    <h3>🔒 Data Usage Consent (DPDP Act)</h3>
                    <p style="font-size: 15px;">
                        This app sends your typed input to <b>Google Gemini (third-party AI)</b>
                        to generate road safety recommendations.<br><br>

                        ✅ Data not stored<br>
                        ✅ User can withdraw at anytime<br>
                        ⚠ Do NOT enter personal or confidential data
                    </p>
            """,
            unsafe_allow_html=True,
        )

        consent = st.checkbox(
            "I agree to share my input with Google Gemini (third-party AI service)."
        )

        if st.button("✅ Accept & Continue"):
            if consent:
                st.session_state.consent_given = True
                st.rerun()
            else:
                st.error("⚠ Please accept consent to continue.")

        st.markdown("</div></div>", unsafe_allow_html=True)


# 🚫 BLOCK APP UNTIL CONSENT GIVEN
if not st.session_state.consent_given:
    consent_popup()
    st.stop()


# ---------------------------------------------------------------------
# 🎨 HEADER WITH EMBEDDED CAR BACKGROUND (cars.png must be in same folder)
# ---------------------------------------------------------------------
def get_base64_image(path):
    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode()

try:
    base64_image = get_base64_image("cars.png")

    st.markdown(f"""
    <style>
        .title-container {{
            background-image: url("data:image/gif;base64,{base64_image}");
            background-size: cover;
            background-position: center;
            height: 120px;
            border-radius: 12px;
            padding: 12px;
            position: relative;
            overflow: hidden;
            margin-bottom: 1rem;
        }}
        .title-overlay {{
            position: absolute;
            background-color: rgba(255,255,255,0.75);
            width: 100%; height: 100%; top: 0; left: 0;
        }}
        .title-content {{
            text-align: center;
            position: relative; z-index: 2;
        }}
        .title-content h1 {{
            color: #FF8F00;
            margin-top: 10px;
            font-size: 1.8rem;
            font-weight: 700;
        }}
    </style>

    <div class="title-container">
        <div class="title-overlay"></div>
        <div class="title-content">
            <h1>🚦 Road Safety Advisor (v4.8)</h1>
            <p>AI-Powered Interventions — DPDP Consent Enabled</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

except:
    st.title("🚦 Road Safety Advisor (v4.8)")
    st.caption("Cars background not found")


# ---------------------------------------------------------------------
# 🔐 GEMINI API KEY INPUT (Manual)
# ---------------------------------------------------------------------
st.sidebar.header("🔐 Gemini AI Configuration")
manual_key = st.sidebar.text_input(
    "Enter your Gemini API Key:",
    type="password",
    placeholder="AIza...."
)

if manual_key:
    os.environ["GEMINI_API_KEY"] = manual_key
    genai.configure(api_key=manual_key)
    gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")
    st.sidebar.success("✅ Gemini Enabled")
else:
    gemini_model = None
    st.sidebar.warning("Gemini API key not set. AI reasoning disabled.")


# ---------------------------------------------------------------------
# 📤 LOAD CSV DATABASE
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
# 🔍 VECTOR SEARCH ENGINE
# ---------------------------------------------------------------------
@st.cache_resource
def build_search_engine(df):
    vect = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf = vect.fit_transform(df["searchable_text"])
    model = NearestNeighbors(metric="cosine").fit(tfidf)
    return vect, model, tfidf

vectorizer, nbrs, tfidf_matrix = build_search_engine(df)


# ---------------------------------------------------------------------
# 🧾 INPUT FORM
# ---------------------------------------------------------------------
st.header("📝 Input Road Safety Problem Details")
safety_issues = st.text_area("⚠ Safety Issue *", height=100)
environmental_conditions = st.text_area("🌍 Environment / Road Conditions *", height=80)
problem_description = st.text_area("🔍 Detailed Problem Description *", height=100)

if st.button("🔎 Identify Suitable Interventions", use_container_width=True):

    if not (safety_issues.strip() and environmental_conditions.strip() and problem_description.strip()):
        st.error("⚠ Please fill all required fields.")
        st.stop()

    query = f"{safety_issues} {environmental_conditions} {problem_description}"
    distances, indices = nbrs.kneighbors(vectorizer.transform([query]), n_neighbors=3)

    st.subheader("✅ Top 3 Recommended Interventions")

    for idx, row_index in enumerate(indices[0]):
        row = df.iloc[row_index]

        st.markdown("---")
        st.subheader(f"#{idx+1} — {row['problem_category']} ({row['type']})")
        st.write(row["data"])

        if gemini_model:
            with st.expander("🤖 AI Explanation"):
                st.write(
                    gemini_model.generate_content(
                        f"Problem: {problem_description}\nIntervention: {row['data']}\nExplain benefit in 5 sentences."
                    ).text
                )

st.caption("🛡 DPDP Consent | AI Reasoning | Map | CSV Export Available")
