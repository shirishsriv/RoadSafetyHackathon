# 🚦 Road Safety Intervention GPT — v4.8

AI-powered **Road Safety Intervention Identification Tool** built using Streamlit + Gemini AI.  
The app takes road safety problem details (hazards, environment, problem description) and automatically recommends the **best matching safety interventions** from a CSV database.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 AI Reasoning (Google Gemini) | Explains *why* the intervention is suitable |
| 🔎 Smart Search Engine | Uses TF-IDF + Nearest Neighbors to match input text to interventions |
| 🗺️ Map Visualization | Plots relevant interventions using latitude/longitude from the dataset |
| 📥 CSV Upload Support | Use your own database of interventions |
| 📤 Export Results | Download recommended interventions as CSV |
| 🎨 UI Enhancements | Light theme with moving car background GIF |

---

## 🚀 Live Demo (Streamlit Cloud)

🔗 **App link:** https://roadsafety-testing.streamlit.app
Example: `https://road-safety-gpt.streamlit.app/`

---

## 🛠️ Tech Stack

- **Frontend/UI** → Streamlit
- **AI Model** → Google Gemini (`google-generativeai`)
- **Search engine** → TF-IDF + Nearest Neighbors (`scikit-learn`)
- **Map Visualization** → Folium + streamlit-folium
- **Database** → CSV-based intervention library

---

## 📦 Installation

Clone the repo:

```bash
git clone https://github.com/<your-username>/road-safety-gpt.git
cd road-safety-gpt
