# Projekt: MovieMate – finaler Movie-Recommender mit UI/UX Verbesserungen

import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import os
import requests
import re
import uuid
import gdown

# === App Setup ===
st.set_page_config(page_title="MovieMate", page_icon="🎬", layout="wide")

# Zufällige User-ID erzeugen
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

# Funktion Titel bereinigen
def clean_title(title):
    return re.sub(r"\s*\(\d{4}\)", "", title).strip()

# Poster abrufen
def get_movie_poster(title, api_key):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": api_key, "query": title}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json().get("results")
        if results:
            poster_path = results[0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None

# Text-Erklärung
def generate_text_explanation(movie_row, tags_selected):
    reasons = []
    genre_sim = movie_row.get("genre_similarity", 0)
    tag_sim = movie_row.get("tag_similarity", 0)
    rating = movie_row.get("avg_rating", 0)

    if genre_sim > 0.65:
        reasons.append("<b>starke Genre-Überschneidungen</b> mit deinen Lieblingsfilmen")
    elif genre_sim > 0.4:
        reasons.append("gewisse <b>inhaltliche Ähnlichkeiten</b> in den Genres")

    if tag_sim > 0.4 and tags_selected:
        reasons.append("viele deiner <b>gewählten Tags</b> werden abgedeckt")
    elif tag_sim > 0.2 and tags_selected:
        reasons.append("einige Schlagworte <b>passen zu deinen Interessen</b>")

    if rating >= 4.0:
        reasons.append("von anderen Nutzer:innen <b>sehr gut bewertet</b>")
    elif rating >= 3.6:
        reasons.append("<b>solide Bewertungen</b> von der Community")

    trust_score = movie_row.get("similarity", 0)
    trust_percent = round(trust_score * 100, 1)
    vertrauen_text = f" 🔒 <b>Vertrauenswert:</b> {trust_percent} %"

    if reasons:
        return "Empfohlen, weil " + " und ".join(reasons) + ". " + vertrauen_text
    else:
        return "Empfohlen, weil er in mehreren Aspekten zu deinem Profil passt. " + vertrauen_text

# === Google Drive CSV-Download ===
def download_and_verify_csv(file_id, dest_path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    if not os.path.exists(dest_path):
        gdown.download(url, dest_path, quiet=False)
    with open(dest_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        if "<html" in first_line.lower():
            st.error(f"❌ Fehler beim Download: '{dest_path}' enthält HTML statt CSV.")
            st.stop()

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

# === UI Startseite ===
st.markdown("<h1 style='text-align:center;'>🎬 MovieMate</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Finde Filme, die perfekt zu deinem Geschmack passen – mit Begründung, warum sie dir gefallen könnten.</p>", unsafe_allow_html=True)
st.markdown("---")

# === Film-Auswahl ===
st.subheader("✨ Deine Auswahl")

min_year = st.slider("Zeige Filme ab Jahr:", 1950, 2015, 1999)
movies = movies[movies["year"] >= min_year]
available_movies = movies.sort_values("title")

selected_titles = []
for i in range(1, 6):
    film = st.selectbox(f"🎥 Wähle Film {i}:", ["-- bitte auswählen --"] + available_movies["title"].tolist(), key=f"film_{i}")
    if film != "-- bitte auswählen --":
        selected_titles.append(film)

tags_selected = []
with st.expander("🔖 Optional: Wähle Tags, die dich interessieren"):
    all_tags = genome_tags["tag"].sort_values().unique().tolist()
    tags_selected = st.multiselect("Bis zu 5 Tags:", all_tags, max_selections=5)

# === Empfehlungen ===
if len(selected_titles) == 5:
    selected_ids = movies[movies["title"].isin(selected_titles)]["movieId"].values
    movie_features = movies.copy()
    movie_features = movie_features.join(movies["genres"].str.get_dummies("|"))
    genre_columns = movies["genres"].str.get_dummies("|").columns
    user_profile = movie_features[movie_features["movieId"].isin(selected_ids)][genre_columns].mean().values.reshape(1, -1)
    all_profiles = movie_features[genre_columns].values
    genre_similarities = cosine_similarity(user_profile, all_profiles)[0]
    movies["genre_similarity"] = genre_similarities

    tag_matrix = pd.pivot_table(genome_scores, values="relevance", index="movieId", columns="tagId", fill_value=0)
    selected_tag_ids = genome_tags[genome_tags["tag"].isin(tags_selected)]["tagId"].tolist()
    user_tag_vector = pd.Series(0, index=tag_matrix.columns, dtype=float)
    for tag_id in selected_tag_ids:
        user_tag_vector[tag_id] = 1.0

    tag_similarities = cosine_similarity([user_tag_vector], tag_matrix.reindex(movies["movieId"].values, fill_value=0).fillna(0).values)[0]
    movies["tag_similarity"] = tag_similarities

    movies["similarity"] = 0.5 * movies["genre_similarity"] + 0.5 * movies["tag_similarity"] if tags_selected else movies["genre_similarity"]
    top_movies = movies[~movies["movieId"].isin(selected_ids)].sort_values("similarity", ascending=False).head(3)

    st.subheader("🌟 Deine Empfehlungen")

    api_key = st.secrets["TMDB_API_KEY"]

    for _, row in top_movies.iterrows():
        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                poster_url = get_movie_poster(clean_title(row["title"]), api_key)
                st.image(poster_url if poster_url else "https://via.placeholder.com/120x180.png?text=No+Image", width=250)

            with col2:
                st.markdown(f"<h3>{row['title']}</h3>", unsafe_allow_html=True)
                explanation = generate_text_explanation(row, tags_selected)
                st.markdown(f"<p style='font-size:16px; color:#333;'>{explanation}</p>", unsafe_allow_html=True)
            st.markdown("---")






