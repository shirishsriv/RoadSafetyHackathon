# ---------------------------------------------------------------------
# 🚦 ROAD SAFETY GPT ADVISOR VERSION 4.8 — LIGHT THEME + CAR BACKGROUND
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
# ⚙️ STREAMLIT CONFIG
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Road Safety Advisor v4.8 — Hackathon Edition",
    page_icon="🚧",
    layout="wide"
)

# ---------------------------------------------------------------------
# 🎨 EMBEDDED CAR BACKGROUND (cars.png)
# ---------------------------------------------------------------------

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

try:
    base64_image = get_base64_image("cars.png")  # Animated car GIF

    st.markdown (
    f"""
    <style>

    /* ========= FIX FOR ERROR VISIBILITY ========= */
    div[data-testid="stNotification"], .stAlert, .st-error {{
        background-color: #ffebee !important;
        color: #b00020 !important;
        border: 2px solid #b00020 !important;
        font-weight: 700 !important;
    }}
    div[data-testid="stNotification"] p {{ color: #b00020 !important; }}

    /* ========= HEADER WITH GIF BACKGROUND ========= */
    .title-container {{
        background-image: url("data:image/gif;base64,{base64_image}");
        background-size: cover;
        background-position: center;
        height: 120px;               /* Reduced header size */
        border-radius: 12px;
        padding: 12px;
        position: relative;
        overflow: hidden;
        margin-bottom: 1rem;
    }}

    /* Transparent overlay so text stays readable */
    .title-overlay {{
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background-color: rgba(255, 255, 255, 0.75);
        z-index: 1;
    }}

    .title-content {{
        position: relative;
        z-index: 2;
        text-align: center;
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
            <p>AI-powered intervention recommendations</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
    )


except FileNotFoundError:
    st.title("🚦 Road Safety Advisor (v4.8)")
    st.caption("AI-Powered Recommendations • Interactive Map • Light Theme 🚗💡")

# ---------------------------------------------------------------------
# 🧠 THREAD-SAFETY SETTINGS
# ---------------------------------------------------------------------
original_stderr = sys.stderr
class FilteredStderr:
    def __init__(self, original): self.original = original
    def write(self, s):
        if any(x in s for x in ["mutex.cc", "RAW: Lock blocking", "Lock blocking 0x"]): return
        self.original.write(s)
    def flush(self): self.original.flush()
    def __getattr__(self, name): return getattr(self.original, name)
sys.stderr = FilteredStderr(original_stderr)

# ---------------------------------------------------------------------
# 🚦 HEADER
# ---------------------------------------------------------------------
st.title("🚦 Road Safety Intervention Identification Tool (v4.8)")
st.caption("Hackathon Edition — Light Theme with Car Background 🏎️")

# ---------------------------------------------------------------------
# 🔐 GEMINI API KEY INPUT
# ---------------------------------------------------------------------
st.sidebar.header("🔐 Gemini AI Configuration")
manual_key = st.sidebar.text_input(
    "Enter your Gemini API key:",
    type="password",
    placeholder="AIza...",
    help="Get your free API key at https://aistudio.google.com/app/apikey"
)
if manual_key:
    os.environ["GEMINI_API_KEY"] = manual_key
    st.sidebar.success("✅ Gemini API key set successfully.")

# ---------------------------------------------------------------------
# 🧾 LOAD DATABASE
# ---------------------------------------------------------------------
@st.cache_data
def load_interventions_database(file_data=None):
    try:
        if file_data:
            df = pd.read_csv(file_data)
        else:
            df = pd.read_csv("interventions.csv", encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv("interventions.csv", encoding="latin1")
    except Exception as e:
        st.error(f"❌ Error loading database: {e}")
        return pd.DataFrame()

    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
    rename_map = {
        "problem": "Problem_Category", "category": "Problem_Category",
        "type": "Type", "data": "Data", "code": "Code", "clause": "Clause"
    }
    df.rename(columns=rename_map, inplace=True)
    required = {"Problem_Category", "Type", "Data"}
    if not required.issubset(df.columns):
        st.error(f"❌ Missing columns: {required - set(df.columns)}")
        return pd.DataFrame()

    df["searchable_text"] = df[["Problem_Category", "Type", "Data"]].astype(str).agg(" ".join, axis=1)
    if "latitude" not in df.columns or "longitude" not in df.columns:
        df["latitude"] = np.nan
        df["longitude"] = np.nan
    return df

# ---------------------------------------------------------------------
# 🔍 SEARCH ENGINE
# ---------------------------------------------------------------------
@st.cache_resource
def create_vectorizer():
    return TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")

@st.cache_resource
def create_search_index(df, _vectorizer=None):
    if df.empty or _vectorizer is None: return None, None, None
    tfidf_matrix = _vectorizer.fit_transform(df["searchable_text"])
    nbrs = NearestNeighbors(n_neighbors=min(20, len(df)), metric="cosine").fit(tfidf_matrix)
    return nbrs, tfidf_matrix, _vectorizer

def search_interventions(query, df, nbrs, tfidf_matrix, vectorizer, k=3, min_similarity=0.3):
    if df.empty or nbrs is None: return pd.DataFrame(), np.array([])
    query_vec = vectorizer.transform([query])
    distances, indices = nbrs.kneighbors(query_vec, n_neighbors=min(k, len(df)))
    valid_indices = [i for i in indices[0] if 0 <= i < len(df)]
    if not valid_indices: return pd.DataFrame(), np.array([])
    results = df.iloc[valid_indices].copy()
    similarity_scores = 1.0 - (distances[0] / 2.0)
    results["similarity_score"] = similarity_scores
    results = results[similarity_scores >= min_similarity]
    return results.sort_values(by="similarity_score", ascending=False), similarity_scores

# ---------------------------------------------------------------------
# 🤖 GEMINI REASONING
# ---------------------------------------------------------------------
def generate_reasoning(problem, issues, conditions, intervention, model):
    prompt = f"""
You are a road safety expert.

Safety Issues: {issues}
Environmental Conditions: {conditions}
Problem: {problem}

Recommended Intervention:
- Category: {intervention.get('Problem_Category')}
- Type: {intervention.get('Type')}
- Description: {intervention.get('Data')}

Explain in 4–5 sentences why this intervention is suitable and effective.
"""
    try:
        response = model.generate_content(prompt)
        return response.text if hasattr(response, "text") else str(response)
    except Exception as e:
        return f"⚠️ Error generating reasoning: {e}"

# ---------------------------------------------------------------------
# 🧠 LOAD GEMINI MODEL
# ---------------------------------------------------------------------
gemini_model = None
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    try:
        genai.configure(api_key=gemini_key)
        gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")
        st.sidebar.success("🤖 Gemini AI reasoning enabled.")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Gemini AI disabled: {e}")
else:
    st.sidebar.info("💡 Enter your API key to enable Gemini AI reasoning.")

# ---------------------------------------------------------------------
# 📤 CSV UPLOAD
# ---------------------------------------------------------------------
st.sidebar.header("📂 Database Options")
uploaded_file = st.sidebar.file_uploader("Upload your interventions CSV", type=["csv"])
if uploaded_file:
    df = load_interventions_database(uploaded_file)
    st.sidebar.success("✅ Custom CSV loaded successfully!")
else:
    df = load_interventions_database()
if df.empty:
    st.stop()

# ---------------------------------------------------------------------
# 🧾 INPUT FORM
# ---------------------------------------------------------------------
st.header("📝 Input Road Safety Problem Details")
safety_issues = st.text_area("⚠️ Safety Issues Identified *", height=100)
environmental_conditions = st.text_area("🌍 Environmental Conditions *", height=80)
problem_description = st.text_area("🔍 Detailed Problem Description *", height=100)
additional_context = st.text_area("📋 Additional Context (Optional)", height=80)

# ---------------------------------------------------------------------
# 🚀 SEARCH & RESULTS
# ---------------------------------------------------------------------
vectorizer = create_vectorizer()
nbrs, tfidf_matrix, _ = create_search_index(df, _vectorizer=vectorizer)

if st.button("🔎 Identify Suitable Interventions", type="primary", use_container_width=True):
    if not (safety_issues.strip() and environmental_conditions.strip() and problem_description.strip()):
        st.warning("⚠️ Please fill all required fields.")
        st.stop()

    query = f"Safety issues: {safety_issues}. Environmental: {environmental_conditions}. Problem: {problem_description}. {additional_context}"
    results, scores = search_interventions(query, df, nbrs, tfidf_matrix, vectorizer, k=3, min_similarity=0.3)

    if results.empty:
        st.warning("⚠️ No matching interventions found.")
        st.stop()

    st.success("✅ Analysis complete! Top 3 interventions displayed below.")
    st.header("🎯 Top 3 Recommended Interventions")

    for idx, (_, row) in enumerate(results.iterrows(), 1):
        st.markdown("---")
        col1, col2 = st.columns([4, 1])
        with col1:
            st.subheader(f"#{idx} {row['Problem_Category']} – {row['Type']}")
        with col2:
            st.metric("Relevance", f"{row['similarity_score']:.1%}")
        st.markdown(f"**📋 Description:** {row['Data']}")
        st.markdown(f"**📚 Reference:** {row.get('Code', 'N/A')} – Clause {row.get('Clause', 'N/A')}")
        if gemini_model:
            with st.expander("🤖 AI-Powered Reasoning", expanded=(idx == 1)):
                st.write(generate_reasoning(problem_description, safety_issues, environmental_conditions, row, gemini_model))

    # ---------------------------------------------------------------------
    # 🗺️ MAP VISUALIZATION
    # ---------------------------------------------------------------------
    if results["latitude"].notna().any() and results["longitude"].notna().any():
        st.subheader("🗺️ Intervention Map")
        valid_points = results.dropna(subset=["latitude", "longitude"])
        avg_lat, avg_lon = valid_points["latitude"].mean(), valid_points["longitude"].mean()
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=6)
        for _, r in valid_points.iterrows():
            folium.Marker(
                [r["latitude"], r["longitude"]],
                popup=f"<b>{r['Problem_Category']}</b><br>{r['Type']}<br>{r['Data']}",
                tooltip=r["Problem_Category"],
                icon=folium.Icon(color="orange", icon="info-sign")
            ).add_to(m)
        st_folium(m, width=700, height=500)
    else:
        st.info("📍 No geographic coordinates found in the database to display a map.")

    # ---------------------------------------------------------------------
    # 💾 EXPORT RESULTS
    # ---------------------------------------------------------------------
    st.markdown("### 💾 Export Recommendations")
    output = io.StringIO()
    results.to_csv(output, index=False)
    st.download_button(
        label="⬇️ Download Results as CSV",
        data=output.getvalue(),
        file_name="recommended_interventions.csv",
        mime="text/csv"
    )

st.caption("**Road Safety Advisor v4.8** — Light Theme + Car Background 🚗💨 | AI Reasoning • Map • Export")
