# MovieMate – Recommender 
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import os, requests, re, uuid, gdown, hashlib
from textwrap import dedent
import openai

st.set_page_config(page_title="MovieMate", page_icon="🎬", layout="wide")

# ----------------- STYLES -----------------
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
  color: #111 !important;
}

.card { background: var(--card-bg); border-radius: 14px; overflow: hidden; box-shadow: 0 6px 14px rgba(0,0,0,.06); margin-bottom: 20px; display: flex; flex-direction: column; align-items: center; }
.card img { width: 100%; height: 260px; object-fit: cover; border-bottom: 1px solid #eee; }
.card__title { font-size: 14px; font-weight: 700; color: #111 !important; height: 44px; display:flex; align-items:center; justify-content:center; text-align:center; overflow:hidden; text-overflow:ellipsis; margin:8px 0; }
.card__explain { font-size: 14px; line-height: 1.6; text-align: left; margin-top: 10px; padding: 0 6px; color: #111 !important; min-height: 250px; max-height: 250px; overflow: hidden; }

.badge { display: inline-block; background: #eef2ff; color: #4338ca; padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 700; margin-bottom: 10px; }
.section-title { margin: 16px 0 12px; font-weight: 800; color: #111 !important; }

.stButton > button { background-color: #e50914 !important; color: #fff !important; font-weight: 700 !important; border: none !important; border-radius: 6px !important; padding: 14px 28px !important; font-size: 16px !important; text-transform: uppercase !important; letter-spacing: .5px !important; box-shadow: 0 6px 20px rgba(229,9,20,.4) !important; transition: background .2s ease, transform .1s ease; width: 100% !important; margin-top: 10px !important; }
.stButton > button:hover { background-color: #f6121d !important; transform: scale(1.03) !important; }
.stButton > button:disabled { opacity: .5 !important; cursor: not-allowed !important; }
</style>
"""), unsafe_allow_html=True)

# ----------------- SESSION STATE -----------------
if "user_id" not in st.session_state: st.session_state.user_id = str(uuid.uuid4())
if "rec_index" not in st.session_state: st.session_state.rec_index = 3
if "selection_key" not in st.session_state: st.session_state.selection_key = None
if "intro_done" not in st.session_state: st.session_state.intro_done = False
if "selected_titles" not in st.session_state: st.session_state.selected_titles = []
if "search_page" not in st.session_state: st.session_state.search_page = 0
if "explanations" not in st.session_state: st.session_state.explanations = {}

# ----------------- HELPERS -----------------
def clean_title(title: str) -> str:
    return re.sub(r"\s*\(\d{4}\)", "", str(title)).strip()

def selection_hash(titles, year_from):
    raw = "|".join(sorted(titles)) + f"||{year_from}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

@st.cache_data(show_spinner=False)
def get_movie_poster(title, api_key):
    if not api_key: return None
    try:
        url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": api_key, "query": title}
        r = requests.get(url, params=params, timeout=8)
        if r.ok:
            results = r.json().get("results", [])
            if results and results[0].get("poster_path"):
                return f"https://image.tmdb.org/t/p/w500{results[0]['poster_path']}"
    except: pass
    return None

openai.api_key = st.secrets.get("OPENAI_API_KEY")
if not openai.api_key:
    st.error("❌ OPENAI_API_KEY fehlt in den Streamlit Secrets.")

# ----------------- DATEN LADEN -----------------
def download_and_verify_csv(file_id, dest_path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    if not os.path.exists(dest_path):
        gdown.download(url, dest_path, quiet=False, use_cookies=False)
    with open(dest_path, "rb") as f:
        head = f.read(4096).lower()
        if b"<html" in head:
            st.error(f"❌ Fehler beim Download: '{dest_path}' enthält HTML statt CSV.")
            st.stop()

def _read_csv_anysep(path):
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8")

os.makedirs("./data", exist_ok=True)
download_and_verify_csv("1AVtktDFEXey1RSTq_lTFE4sgG-S9nIxT","./data/movies.csv")
download_and_verify_csv("17USu4Dkt0SaoL8XiV3ckm1wX2iP7HgQQ","./data/ratings.csv")

@st.cache_data
def load_data():
    base = "./data/"
    movies = _read_csv_anysep(base+"movies.csv")
    ratings = _read_csv_anysep(base+"ratings.csv")
    if "year" not in movies.columns:
        movies["year"] = movies["title"].astype(str).str.extract(r"\((\d{4})\)")
    movies["year"] = pd.to_numeric(movies["year"], errors="coerce").fillna(0).astype(int)
    if {"movieId","rating"}.issubset(ratings.columns):
        agg = ratings.groupby("movieId")["rating"].agg(avg_rating="mean", n_ratings="count")
        movies = movies.merge(agg, on="movieId", how="left")
        movies["avg_rating"] = movies["avg_rating"].fillna(0)
        movies["n_ratings"] = movies["n_ratings"].fillna(0)
        movies = movies[(movies["avg_rating"] >= 3) & (movies["n_ratings"] >= 50)]
    return movies.reset_index(drop=True), ratings

movies, ratings = load_data()

# ----------------- HEADER -----------------
st.markdown("<h1 style='text-align:center;'>🎬 MovieMate</h1>", unsafe_allow_html=True)

# ----------------- INTRO -----------------
if not st.session_state.intro_done:
    hero_html = dedent("""
    <div class="hero">
      <div class="hero__bg"></div>
      <div class="hero__content">
        <div class="hero__title">Willkommen bei MovieMate</div>
        <div class="hero__subtitle">Wähle 5 Filme aus und erhalte deine Empfehlungen.</div>
      </div>
    </div>
    """)
    st.markdown(hero_html, unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        if st.button("🎬 Los geht's", use_container_width=True):
            st.session_state.intro_done = True
            st.rerun()

# ----------------- MAIN -----------------
else:
    min_year = st.slider("Zeige Filme ab Jahr:", 1950, 2015, 1999)

    # --- AUSWAHLPHASE ---
    if len(st.session_state.selected_titles) < 5:
        search = st.text_input("🔎 Film suchen oder aus Liste wählen:", placeholder="Titel eingeben...")
        movies_view = movies[movies["year"] >= min_year].copy()
        available_movies = movies_view.sort_values("title")

        if search:
            mask = available_movies["title"].str.contains(search, case=False, na=False, regex=False)
            available_movies = available_movies[mask].copy()
            available_movies["starts"] = available_movies["title"].str.lower().str.startswith(search.lower())
            available_movies = available_movies.sort_values(by=["starts","title"], ascending=[False, True]).drop(columns=["starts"])

        page_size = 15
        total_pages = max(1, (len(available_movies)-1)//page_size +1)
        start = st.session_state.search_page * page_size
        end = start + page_size
        page_movies = available_movies.iloc[start:end]

        # Filme in 5 Spalten
        for i in range(0, len(page_movies), 5):
            cols = st.columns(5)
            for j, (_, row) in enumerate(page_movies.iloc[i:i+5].iterrows()):
                with cols[j]:
                    api_key = st.secrets.get("TMDB_API_KEY")
                    poster = get_movie_poster(clean_title(row["title"]), api_key) if api_key else None
                    poster = poster or "https://via.placeholder.com/300x450.png?text=No+Image"
                    st.markdown(f"<div class='card'><img src='{poster}'><div class='card__title'>{row['title']}</div></div>", unsafe_allow_html=True)
                    
                    # Auswählen / Entfernen Button
                    is_selected = row["title"] in st.session_state.selected_titles
                    label = "✅ Entfernen" if is_selected else "➕ Auswählen"
                    if st.button(label, key=f"btn_{row['movieId']}"):
                        if is_selected:
                            st.session_state.selected_titles.remove(row["title"])
                        elif len(st.session_state.selected_titles) < 5:
                            st.session_state.selected_titles.append(row["title"])
                        st.rerun()

        # Zentrierter "Mehr Filme laden" Button
        c1,c2,c3 = st.columns([1,2,1])
        with c2:
            if st.button("🔄 Mehr Filme laden", use_container_width=True):
                st.session_state.search_page += 1
                st.rerun()

        st.progress(len(st.session_state.selected_titles)/5)
        st.write(f"Ausgewählt: {len(st.session_state.selected_titles)}/5 Filme")

    # --- EMPFEHLUNGSPHASE ---
    else:
        st.success("✅ Du hast 5 Filme ausgewählt – hier deine Empfehlungen:")
        sel_key = selection_hash(st.session_state.selected_titles, int(min_year))
        if st.session_state.selection_key != sel_key:
            st.session_state.selection_key = sel_key
            st.session_state.rec_index = 3

        selected_ids = movies.loc[movies["title"].isin(st.session_state.selected_titles), "movieId"].dropna().astype(int).values
        genres_full = movies["genres"].astype(str).str.get_dummies("|")
        movie_features_full = movies.join(genres_full)
        user_rows = movie_features_full[movie_features_full["movieId"].isin(selected_ids)]
        genre_cols = genres_full.columns
        user_profile = user_rows[genre_cols].mean(axis=0)

        if user_profile.isna().all():
            st.warning("Kein Profil berechenbar – bitte andere Filme wählen."); st.stop()

        movies_view_rec = movies[movies["year"] >= min_year].copy()
        view_genres = movies_view_rec["genres"].astype(str).str.get_dummies("|")
        for c in genre_cols:
            if c not in view_genres.columns: view_genres[c] = 0
        view_genres = view_genres[genre_cols]
        movies_view_rec["similarity"] = cosine_similarity(user_profile.values.reshape(1,-1), view_genres.values)[0]
        sorted_movies = movies_view_rec.loc[~movies_view_rec["movieId"].isin(selected_ids)].sort_values("similarity", ascending=False).reset_index(drop=True)
        
        max_n = len(sorted_movies)
        show_n = min(st.session_state.rec_index, max_n)
        to_show = sorted_movies.iloc[:show_n]

        st.markdown("<h3 class='section-title'>🌟 Empfehlungen</h3>", unsafe_allow_html=True)
        cols = st.columns(3)
        api_key = st.secrets.get("TMDB_API_KEY")
        for idx, row in to_show.iterrows():
            col = cols[idx % 3]
            with col:
                poster = get_movie_poster(clean_title(row["title"]), api_key) if api_key else None
                poster = poster or "https://via.placeholder.com/500x750.png?text=No+Image"
                if row["movieId"] in st.session_state.explanations:
                    exp = st.session_state.explanations[row["movieId"]]
                else:
                    exp = f"Empfehlung basierend auf deinen ausgewählten Filmen."
                    st.session_state.explanations[row["movieId"]] = exp
                st.markdown(f"<div class='card'><img src='{poster}'><div class='card__body'><div class='badge'>Empfehlung</div><div class='card__title'>{row['title']}</div><div class='card__explain'>{exp}</div></div></div>", unsafe_allow_html=True)

        # Mehr Empfehlungen Button
        can_more = show_n < max_n
        c1,c2,c3 = st.columns([1,2,1])
        with c2:
            if st.button("🔄 Mehr Empfehlungen laden", disabled=not can_more, use_container_width=True):
                st.session_state.rec_index = min(st.session_state.rec_index + 3, max_n)
                st.rerun()
