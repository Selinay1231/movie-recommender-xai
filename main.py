# Projekt: MovieMate – Recommender mit Intro-Flow (Landing Page + "Los geht's")

import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import os
import requests
import re
import uuid
import gdown
import random
import hashlib

# =========================
# App Setup
# =========================
st.set_page_config(page_title="MovieMate", page_icon="🎬", layout="wide")

# Session: User-ID
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

# Session: Anzeige-Index (wie viele Empfehlungen zeigen)
if "rec_index" not in st.session_state:
    st.session_state.rec_index = 3  # immer in 3er-Schritten

# Session: Auswahl-Hash (um bei Änderungen zurückzusetzen)
if "selection_key" not in st.session_state:
    st.session_state.selection_key = None

# Session: Intro-Screen
if "intro_done" not in st.session_state:
    st.session_state.intro_done = False

# =========================
# Hilfsfunktionen
# =========================
def clean_title(title):
    return re.sub(r"\s*\(\d{4}\)", "", title).strip()

def get_movie_poster(title, api_key):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": api_key, "query": title}
    try:
        r = requests.get(url, params=params, timeout=8)
        if r.status_code == 200:
            results = r.json().get("results", [])
            if results and results[0].get("poster_path"):
                return f"https://image.tmdb.org/t/p/w500{results[0]['poster_path']}"
    except Exception:
        pass
    return None

def generate_text_explanation(movie_row, tags_selected):
    reasons = []

    genre_sim = movie_row.get("genre_similarity", 0)
    tag_sim = movie_row.get("tag_similarity", 0)
    rating = movie_row.get("avg_rating", 0)
    year = int(movie_row.get("year", 0)) if not pd.isna(movie_row.get("year", 0)) else None
    n_ratings = movie_row.get("n_ratings", 0)

    # Genre
    if genre_sim > 0.65:
        reasons.append(random.choice([
            "weil er sehr ähnliche Genres hat wie deine Lieblingsfilme",
            "da er thematisch stark an deine bevorzugten Genres anknüpft",
            "weil er inhaltlich fast deckungsgleich mit deinen Genre-Präferenzen ist"
        ]))
    elif genre_sim > 0.4:
        reasons.append(random.choice([
            "weil er teilweise ähnliche Genre-Muster aufweist",
            "da sich bestimmte Themen mit deinen bisherigen Filmen überschneiden",
            "weil er einige typische Elemente deiner Genres enthält"
        ]))

    # Tags
    if tag_sim > 0.4 and tags_selected:
        reasons.append(random.choice([
            "weil er viele deiner gewählten Schlagworte aufgreift",
            "da er stark mit den von dir markierten Themen übereinstimmt",
            "weil die gewählten Tags hier deutlich vertreten sind"
        ]))
    elif tag_sim > 0.2 and tags_selected:
        reasons.append(random.choice([
            "weil er in Teilen zu deinen gewählten Tags passt",
            "da einige Themen mit deinen Interessen übereinstimmen",
            "weil einzelne Schlagworte aus deinen Präferenzen enthalten sind"
        ]))

    # Bewertung
    if rating >= 4.0:
        reasons.append(random.choice([
            "weil er von anderen Nutzer:innen besonders gut bewertet wurde",
            "da er eine außergewöhnlich hohe Durchschnittsbewertung hat",
            "weil er allgemein als sehr sehenswert gilt"
        ]))
    elif rating >= 3.6:
        reasons.append(random.choice([
            "weil er solide und überdurchschnittliche Bewertungen bekommen hat",
            "da viele Zuschauer:innen ihn als gut eingestuft haben",
            "weil er von der Community als empfehlenswert angesehen wird"
        ]))

    # Popularität
    if n_ratings >= 5000:
        reasons.append("weil er extrem beliebt ist und von vielen Menschen gesehen wurde")
    elif n_ratings >= 1000:
        reasons.append("weil er eine beachtliche Anzahl an Bewertungen erhalten hat")

    # Jahr
    if year and year > 2010:
        reasons.append("weil er ein relativ neuer Film ist, der moderne Themen aufgreift")
    elif year and year < 2000:
        reasons.append("weil er ein Klassiker ist, der bis heute relevant geblieben ist")

    # Vertrauenswert
    trust_score = movie_row.get("similarity", 0)
    trust_percent = round(trust_score * 100, 1)
    if trust_score >= 0.8:
        trust_label = "sehr hoch"
    elif trust_score >= 0.6:
        trust_label = "hoch"
    elif trust_score >= 0.4:
        trust_label = "mittel"
    else:
        trust_label = "niedrig"

    vertrauen_text = f"🔒 <b>Vertrauenswert:</b> {trust_percent} % ({trust_label})"

    if reasons:
        return "Dieser Film wurde empfohlen, " + " und ".join(reasons) + ". " + vertrauen_text
    else:
        return "Dieser Film passt in mehreren Aspekten zu deinem Profil. " + vertrauen_text

def download_and_verify_csv(file_id, dest_path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    if not os.path.exists(dest_path):
        gdown.download(url, dest_path, quiet=False)
    with open(dest_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        if "<html" in first_line.lower():
            st.error(f"❌ Fehler beim Download: '{dest_path}' enthält HTML statt CSV.")
            st.stop()

def selection_hash(titles, tags, year_from):
    raw = "|".join(sorted(titles)) + "||" + "|".join(sorted(tags)) + f"||{year_from}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

# =========================
# Daten laden
# =========================
os.makedirs("./data", exist_ok=True)
download_and_verify_csv("1AVtktDFEXey1RSTq_lTFE4sgG-S9nIxT", "./data/movies.csv")
download_and_verify_csv("17USu4Dkt0SaoL8XiV3ckm1wX2iP7HgQQ", "./data/ratings.csv")
download_and_verify_csv("1wwWoz4RI9ysYVe5mtqNh7BBJ5JwL9IZj", "./data/genome-tags.csv")
download_and_verify_csv("1M0v8mSSbgS7Wz1HoMdCM_YqpXTh0bGd9", "./data/genome-scores.csv")

@st.cache_data
def load_data():
    base_path = "./data/"
    movies = pd.read_csv(base_path + "movies.csv", sep=";", encoding="utf-8")
    ratings = pd.read_csv(base_path + "ratings.csv", sep=";", encoding="utf-8")

    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)").astype(float)
    avg_ratings = ratings.groupby("movieId")["rating"].mean()
    count_ratings = ratings.groupby("movieId")["rating"].count()
    movies = movies.join(avg_ratings.rename("avg_rating"), on="movieId")
    movies = movies.join(count_ratings.rename("n_ratings"), on="movieId")
    movies = movies[(movies["avg_rating"] >= 3) & (movies["n_ratings"] >= 50)]
    return movies.reset_index(drop=True), ratings

@st.cache_data
def load_tag_data():
    base_path = "./data/"
    genome_tags = pd.read_csv(base_path + "genome-tags.csv", sep=";", encoding="utf-8")
    genome_scores = pd.read_csv(base_path + "genome-scores.csv", sep=";", encoding="utf-8")
    return genome_tags, genome_scores

movies, ratings = load_data()
genome_tags, genome_scores = load_tag_data()

# =========================
# UI
# =========================
st.markdown("<h1 style='text-align:center;'>🎬 MovieMate</h1>", unsafe_allow_html=True)

if not st.session_state.intro_done:
    st.markdown(
        """
        <div style='background-color:#f0f2f6; padding:25px; border-radius:12px; text-align:center;'>
            <h2>Willkommen bei MovieMate</h2>
            <p>Unsere KI findet Filme, die perfekt zu deinem Geschmack passen.</p>
            <p>Wähle einfach 5 Filme, die du magst, und erhalte Empfehlungen mit Erklärung.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Button CSS
    st.markdown(
        """
        <style>
        div.stButton {
            display: flex;
            justify-content: center;
        }
        div.stButton > button:first-child {
            background-color: #1f77b4;
            color: white;
            padding: 15px 40px;
            font-size: 20px;
            border-radius: 10px;
            border: none;
            cursor: pointer;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if st.button("🎬 Los geht's"):
        st.session_state.intro_done = True
        st.rerun()

else:
    # =========================
    # Haupt-Recommender UI
    # =========================
    st.subheader("✨ Deine Auswahl")

    min_year = st.slider("Zeige Filme ab Jahr:", 1950, 2015, 1999)
    movies_view = movies[movies["year"] >= min_year]
    available_movies = movies_view.sort_values("title")

    selected_titles = []
    for i in range(1, 6):
        film = st.selectbox(
            f"🎥 Wähle Film {i}:",
            ["-- bitte auswählen --"] + available_movies["title"].tolist(),
            key=f"film_{i}"
        )
        if film != "-- bitte auswählen --":
            selected_titles.append(film)

    tags_selected = []
    with st.expander("🔖 Optional: Tags auswählen"):
        all_tags = genome_tags["tag"].sort_values().unique().tolist()
        tags_selected = st.multiselect("Bis zu 5 Tags:", all_tags, max_selections=5)

    # Empfehlungen anzeigen
    if len(selected_titles) == 5:
        sel_key = selection_hash(selected_titles, tags_selected, int(min_year))
        if st.session_state.selection_key != sel_key:
            st.session_state.selection_key = sel_key
            st.session_state.rec_index = 3

        selected_ids = movies_view[movies_view["title"].isin(selected_titles)]["movieId"].values
        movie_features = movies_view.copy()
        movie_features = movie_features.join(movies_view["genres"].str.get_dummies("|"))
        genre_columns = movies_view["genres"].str.get_dummies("|").columns
        user_profile = movie_features[movie_features["movieId"].isin(selected_ids)][genre_columns].mean().values.reshape(1, -1)
        all_profiles = movie_features[genre_columns].values
        genre_similarities = cosine_similarity(user_profile, all_profiles)[0]
        movies_view["genre_similarity"] = genre_similarities

        tag_matrix = pd.pivot_table(genome_scores, values="relevance", index="movieId", columns="tagId", fill_value=0)
        selected_tag_ids = genome_tags[genome_tags["tag"].isin(tags_selected)]["tagId"].tolist()
        user_tag_vector = pd.Series(0, index=tag_matrix.columns, dtype=float)
        for tag_id in selected_tag_ids:
            user_tag_vector[tag_id] = 1.0

        tag_similarities = cosine_similarity(
            [user_tag_vector],
            tag_matrix.reindex(movies_view["movieId"].values, fill_value=0).fillna(0).values
        )[0]
        movies_view["tag_similarity"] = tag_similarities

        movies_view["similarity"] = (
            0.5 * movies_view["genre_similarity"] + 0.5 * movies_view["tag_similarity"]
            if tags_selected else movies_view["genre_similarity"]
        )
        sorted_movies = movies_view[~movies_view["movieId"].isin(selected_ids)] \
            .sort_values("similarity", ascending=False) \
            .reset_index(drop=True)

        max_n = len(sorted_movies)
        show_n = min(st.session_state.rec_index, max_n)
        to_show = sorted_movies.iloc[:show_n]

        st.subheader("🌟 Deine Empfehlungen")
        api_key = st.secrets.get("TMDB_API_KEY", None)

        for _, row in to_show.iterrows():
            with st.container():
                st.markdown(
                    f"""
                    <div style="background-color:#f9f9f9; padding:20px; margin-bottom:20px;
                                border-radius:12px; box-shadow:0 2px 8px rgba(0,0,0,0.15);">
                        <h3>🎥 {row['title']}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                col1, col2 = st.columns([1, 3])
                with col1:
                    poster_url = get_movie_poster(clean_title(row["title"]), api_key) if api_key else None
                    st.image(poster_url if poster_url else "https://via.placeholder.com/200x300.png?text=No+Image", width=200)
                with col2:
                    explanation = generate_text_explanation(row, tags_selected)
                    st.markdown(f"<p style='font-size:16px; color:#333;'>{explanation}</p>", unsafe_allow_html=True)
                st.markdown("---")

        more_possible = show_n < max_n
        if st.button("🔄 Mehr Empfehlungen laden", disabled=not more_possible):
            st.session_state.rec_index = min(st.session_state.rec_index + 3, max_n)
            st.rerun()



