# MovieMate – eleganter Movie-Recommender (Hero Landing + Grid Cards + Fix Farben + Grid 3x)

import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import os, requests, re, uuid, gdown, random, hashlib

# =========================
# App Setup & Theme
# =========================
st.set_page_config(page_title="MovieMate", page_icon="🎬", layout="wide")

# Global CSS
st.markdown("""
<style>
:root{
  --primary:#6c5ce7; --primary-dark:#5a4bd6;
  --bg-soft:#f4f6fb; --card-bg:#ffffff; --muted:#6b7280;
}
html, body, [data-testid="stApp"] { background: var(--bg-soft); }

/* Headline */
h1 {
  font-weight: 800;
  letter-spacing: .3px;
  color: #111 !important;   /* immer schwarz */
}

/* Buttons */
div.stButton { display:flex; justify-content:center; }
div.stButton > button:first-child{
  background: var(--primary);
  color:#fff; border:none; border-radius:12px;
  padding:14px 28px; font-size:18px; font-weight:600;
  box-shadow:0 6px 20px rgba(108,92,231,.25);
  transition:transform .06s ease, background .2s ease;
}
div.stButton > button:first-child:hover{ background: var(--primary-dark); transform: translateY(-1px); }
div.stButton > button:first-child:disabled{ opacity:.45; cursor:not-allowed; }

/* Hero */
.hero {
  position: relative;
  border-radius: 18px;
  overflow: hidden;
  box-shadow: 0 10px 40px rgba(0,0,0,.08);
  margin-top: 8px;
}
.hero__bg {
  background-image: url('https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?auto=format&fit=crop&w=1600&q=80');
  background-size: cover;
  background-position: center;
  height: 290px;
}
.hero__content {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 0 24px;
  background: rgba(0,0,0,0.35);
}
.hero__title {
  font-size: clamp(24px, 6vw, 44px);
  font-weight: 800;
  margin: 0 0 4px;
  color: #fff !important;
}
.hero__subtitle {
  font-size: clamp(14px, 4vw, 18px);
  margin: 8px 0 0;
  color: #fff !important;
  opacity: .95;
}

/* Cards */
.grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);  /* exakt 3 pro Reihe */
  gap: 20px;
}
.card {
  background: var(--card-bg);
  border-radius: 14px;
  overflow: hidden;
  box-shadow: 0 8px 20px rgba(0,0,0,.06);
  transition: transform .08s ease, box-shadow .2s ease;
}
.card:hover {
  transform: translateY(-3px);
  box-shadow: 0 12px 28px rgba(0,0,0,.12);
}
.card img {
  width: 100%;
  height: 300px;
  object-fit: cover;
  border-bottom: 1px solid #eee;
  background: #e5e7eb;
}
.card__body { padding: 14px 16px 18px; }
.card__title {
  margin: 0 0 8px;
  font-size: 17px;
  font-weight: 700;
  color: #111 !important;
}
.card__explain {
  color: #374151 !important;
  line-height: 1.45;
  font-size: 15px;
}
.badge {
  display: inline-block;
  background: #eef2ff;
  color: #4338ca;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
  margin-bottom: 8px;
}
.section-title {
  margin: 10px 0 8px;
  font-weight: 800;
  letter-spacing: .2px;
  color: #111 !important;
}

/* Widget Labels & Text */
label, .stSelectbox label, .stMultiSelect label, .stSlider label {
  color: #111 !important;
  font-weight: 600;
}
.stSelectbox div[data-baseweb="select"] > div { color: #111 !important; }
.stMultiSelect div[data-baseweb="select"] > div { color: #111 !important; }
</style>
""", unsafe_allow_html=True)

# =========================
# Session State
# =========================
if "user_id" not in st.session_state: st.session_state.user_id = str(uuid.uuid4())
if "rec_index" not in st.session_state: st.session_state.rec_index = 3
if "selection_key" not in st.session_state: st.session_state.selection_key = None
if "intro_done" not in st.session_state: st.session_state.intro_done = False

# =========================
# Helpers
# =========================
def clean_title(title): return re.sub(r"\s*\(\d{4}\)", "", title).strip()

def get_movie_poster(title, api_key):
    url="https://api.themoviedb.org/3/search/movie"; params={"api_key": api_key, "query": title}
    try:
        r=requests.get(url, params=params, timeout=8)
        if r.status_code==200:
            results=r.json().get("results",[])
            if results and results[0].get("poster_path"):
                return f"https://image.tmdb.org/t/p/w500{results[0]['poster_path']}"
    except Exception: pass
    return None

def generate_text_explanation(movie_row, tags_selected):
    reasons=[]
    genre_sim=movie_row.get("genre_similarity",0); tag_sim=movie_row.get("tag_similarity",0)
    rating=movie_row.get("avg_rating",0); year=int(movie_row.get("year",0)) if not pd.isna(movie_row.get("year",0)) else None
    n_ratings=movie_row.get("n_ratings",0)

    if genre_sim>0.65:
        reasons.append(random.choice([
            "weil er sehr ähnliche Genres hat wie deine Lieblingsfilme",
            "da er thematisch stark an deine bevorzugten Genres anknüpft",
            "weil er inhaltlich fast deckungsgleich mit deinen Genre-Präferenzen ist"]))
    elif genre_sim>0.4:
        reasons.append(random.choice([
            "weil er teilweise ähnliche Genre-Muster aufweist",
            "da sich bestimmte Themen mit deinen bisherigen Filmen überschneiden",
            "weil er einige typische Elemente deiner Genres enthält"]))

    if tag_sim>0.4 and tags_selected:
        reasons.append(random.choice([
            "weil er viele deiner gewählten Schlagworte aufgreift",
            "da er stark mit den von dir markierten Themen übereinstimmt",
            "weil die gewählten Tags hier deutlich vertreten sind"]))
    elif tag_sim>0.2 and tags_selected:
        reasons.append(random.choice([
            "weil er in Teilen zu deinen gewählten Tags passt",
            "da einige Themen mit deinen Interessen übereinstimmen",
            "weil einzelne Schlagworte aus deinen Präferenzen enthalten sind"]))

    if rating>=4.0:
        reasons.append(random.choice([
            "weil er von anderen Nutzer:innen besonders gut bewertet wurde",
            "da er eine außergewöhnlich hohe Durchschnittsbewertung hat",
            "weil er allgemein als sehr sehenswert gilt"]))
    elif rating>=3.6:
        reasons.append(random.choice([
            "weil er solide und überdurchschnittliche Bewertungen bekommen hat",
            "da viele Zuschauer:innen ihn als gut eingestuft haben",
            "weil er von der Community als empfehlenswert angesehen wird"]))

    if n_ratings>=5000: reasons.append("weil er extrem beliebt ist und von vielen Menschen gesehen wurde")
    elif n_ratings>=1000: reasons.append("weil er eine beachtliche Anzahl an Bewertungen erhalten hat")

    if year and year>2010: reasons.append("weil er ein relativ neuer Film ist, der moderne Themen aufgreift")
    elif year and year<2000: reasons.append("weil er ein Klassiker ist, der bis heute relevant geblieben ist")

    trust=movie_row.get("similarity",0); trust_percent=round(trust*100,1)
    trust_label="sehr hoch" if trust>=0.8 else "hoch" if trust>=0.6 else "mittel" if trust>=0.4 else "niedrig"
    vt=f"🔒 Vertrauenswert: {trust_percent}% ({trust_label})"

    return ("Dieser Film wurde empfohlen, "+ " und ".join(reasons) + ". " + vt) if reasons else ("Dieser Film passt in mehreren Aspekten zu deinem Profil. " + vt)

