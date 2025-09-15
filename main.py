# MovieMate – eleganter Movie-Recommender (Searchbar + Grid Auswahl + Pagination)

import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import os, requests, re, uuid, gdown, random, hashlib
from textwrap import dedent

# =========================
# App Setup & Theme
# =========================
st.set_page_config(page_title="MovieMate", page_icon="🎬", layout="wide")

# Global CSS
st.markdown(dedent("""
<style>
:root{
  --primary:#6c5ce7; --primary-dark:#5a4bd6;
  --bg-soft:#f4f6fb; --card-bg:#ffffff; --muted:#6b7280;
}
html, body, [data-testid="stApp"] { background: var(--bg-soft); }

h1 {
  font-family: 'Montserrat Alternates', sans-serif !important;
  font-weight: 800 !important;
  letter-spacing: 1px;
  color: #111 !important;
}

/* Eingabefelder */
.stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"],
.stTextInput input {
  background: #fff !important;
  color: #111 !important;
  border: 1px solid #ccc !important;
  border-radius: 6px !important;
}

/* Buttons */
div.stButton { display:flex; justify-content:center; }
div.stButton > button:first-child {
  background: #e50914;
  color: #fff;
  border: none;
  border-radius: 4px;
  padding: 12px 24px;
  font-size: 16px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .5px;
  box-shadow: 0 6px 20px rgba(229,9,20,.4);
  transition: background .2s ease, transform .1s ease;
}
div.stButton > button:first-child:hover {
  background: #f6121d;
  transform: scale(1.03);
}

/* Grid */
.grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 20px; }
.card {
  background: var(--card-bg); border-radius: 10px; overflow: hidden;
  box-shadow: 0 4px 12px rgba(0,0,0,.08);
  transition: transform .08s ease;
}
.card:hover { transform: translateY(-3px); }
.card img { width: 100%; height: 250px; object-fit: cover; }
.card__body { padding: 10px; text-align:center; }
.card__title { font-size: 14px; font-weight: 600; color: #111 !important; }
</style>
"""), unsafe_allow_html=True)

# =========================
# Session State
# =========================
if "user_id" not in st.session_state: st.session_state.user_id = str(uuid.uuid4())
if "rec_index" not in st.session_state: st.session_state.rec_index = 3
if "selection_key" not in st.session_state: st.session_state.selection_key = None
if "intro_done" not in st.session_state: st.session_state.intro_done = False
if "selected_titles" not in st.session_state: st.session_state.selected_titles = []
if "search_page" not in st.session_state: st.session_state.search_page = 0

# =========================
# Helpers
# =========================
def clean_title(title): return re.sub(r"\s*\(\d{4}\)", "", str(title)).strip()

def get_movie_poster(title, api_key):
    if not api_key: return None
    try:
        url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": api_key, "query": title}
        r = requests.get(url, params=params, timeout=8)
        if r.status_code == 200:
            results = r.json().get("results", [])
            if results and results[0].get("poster_path"):
                return f"https://image.tmdb.org/t/p/w500{results[0]['poster_path']}"
    except Exception:
        return None
    return None

def generate_text_explanation(row, tags_selected):
    return f"Wir empfehlen dir **{row['title']}**, da er ähnliche Genres hat wie deine Auswahl."

def download_and_verify_csv(file_id, dest_path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    if not os.path.exists(dest_path): gdown.download(url, dest_path, quiet=False)
    with open(dest_path, "rb") as f:
        head = f.read(4096).lower()
        if b"<html" in head:
            st.error(f"❌ Fehler beim Download: '{dest_path}' enthält HTML statt CSV.")
            st.stop()

def selection_hash(titles, tags, year_from):
    raw = "|".join(sorted(titles)) + "||" + "|".join(sorted(tags)) + f"||{year_from}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

# =========================
# Daten laden
# =========================
def _read_csv_anysep(path):
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8")

os.makedirs("./data", exist_ok=True)
download_and_verify_csv("1AVtktDFEXey1RSTq_lTFE4sgG-S9nIxT","./data/movies.csv")
download_and_verify_csv("17USu4Dkt0SaoL8XiV3ckm1wX2iP7HgQQ","./data/ratings.csv")
download_and_verify_csv("1wwWoz4RI9ysYVe5mtqNh7BBJ5JwL9IZj","./data/genome-tags.csv")
download_and_verify_csv("1M0v8mSSbgS7Wz1HoMdCM_YqpXTh0bGd9","./data/genome-scores.csv")

@st.cache_data
def load_data():
    base = "./data/"
    movies = _read_csv_anysep(base+"movies.csv")
    ratings = _read_csv_anysep(base+"ratings.csv")
    genome_tags = _read_csv_anysep(base+"genome-tags.csv")
    genome_scores = _read_csv_anysep(base+"genome-scores.csv")

    if "year" not in movies.columns:
        movies["year"] = movies["title"].astype(str).str.extract(r"\((\d{4})\)")
    movies["year"] = pd.to_numeric(movies["year"], errors="coerce").fillna(0).astype(int)

    agg = ratings.groupby("movieId")["rating"].agg(avg_rating="mean", n_ratings="count")
    movies = movies.merge(agg, on="movieId", how="left")
    movies = movies[(movies["avg_rating"].fillna(0) >= 3) & (movies["n_ratings"].fillna(0) >= 50)]

    return movies.reset_index(drop=True), ratings, genome_tags, genome_scores

movies, ratings, genome_tags, genome_scores = load_data()

# =========================
# UI
# =========================
st.markdown("<h1 style='text-align:center;'>🎬 MovieMate</h1>", unsafe_allow_html=True)

# ---------- INTRO ----------
if not st.session_state.intro_done:
    hero_html = dedent("""
    <div class="hero">
      <div class="hero__bg"></div>
      <div class="hero__content">
        <div class="hero__title">Willkommen bei MovieMate</div>
        <div class="hero__subtitle">Finde Filme, die perfekt zu deinem Geschmack passen.</div>
        <div class="hero__subtitle" style="margin-top:6px;">Wähle einfach 5 Filme über die Suche aus.</div>
      </div>
    </div>
    """)
    st.markdown(hero_html, unsafe_allow_html=True)

    st.write("")
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        if st.button("🎬 Los geht's", use_container_width=True):
            st.session_state.intro_done = True
            st.rerun()

# ---------- MAIN ----------
else:
    st.markdown("<h3 class='section-title'>✨ Deine Auswahl</h3>", unsafe_allow_html=True)

    min_year = st.slider("Zeige Filme ab Jahr:", 1950, 2015, 1999)

    search = st.text_input("🔎 Film suchen oder aus Liste wählen:")

    movies_view = movies[movies["year"] >= min_year].copy()
    available_movies = movies_view.sort_values("title")

    if search:
        available_movies = available_movies[available_movies["title"].str.contains(search, case=False, na=False)]

    # Pagination
    page_size = 25
    total_pages = max(1, (len(available_movies) - 1) // page_size + 1)
    start = st.session_state.search_page * page_size
    end = start + page_size
    page_movies = available_movies.iloc[start:end]

    # Grid Ansicht
    cols = st.columns(5)
    for idx, row in page_movies.iterrows():
        col = cols[idx % 5]
        with col:
            poster = get_movie_poster(clean_title(row["title"]), st.secrets.get("TMDB_API_KEY"))
            poster = poster or "https://via.placeholder.com/200x300.png?text=No+Image"
            is_selected = row["title"] in st.session_state.selected_titles

            if st.button(f"{'✅' if is_selected else '🎥'} {row['title']}", key=f"btn_{row['movieId']}"):
                if is_selected:
                    st.session_state.selected_titles.remove(row["title"])
                elif len(st.session_state.selected_titles) < 5:
                    st.session_state.selected_titles.append(row["title"])
            st.image(poster, use_container_width=True)

    # Pagination Controls
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        if st.button("⬅️ Zurück", disabled=st.session_state.search_page == 0):
            st.session_state.search_page -= 1
            st.rerun()
    with col3:
        if st.button("➡️ Weiter", disabled=st.session_state.search_page >= total_pages - 1):
            st.session_state.search_page += 1
            st.rerun()

    st.progress(len(st.session_state.selected_titles) / 5)
    st.write(f"Ausgewählt: {len(st.session_state.selected_titles)}/5 Filme")

    # Empfehlungen erst nach 5 gewählten Filmen
    if len(st.session_state.selected_titles) == 5:
        st.success("✅ Du hast 5 Filme ausgewählt – Empfehlungen werden berechnet …")
        # → hier bleibt deine bisherige Empfehlungslogik wie gehabt







