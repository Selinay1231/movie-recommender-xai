# Projekt: Finaler Movie-Recommender (nur mit Text-Erklärungen)

import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os
import requests
import re
import uuid
import gdown

# === Hilfsfunktionen ===

# Zufällige, einmalige User-ID erzeugen (in Session gespeichert)
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

# Funktion zum Bereinigen des Filmtitels (Klammern mit Jahr entfernen)
def clean_title(title):
    return re.sub(r"\s*\(\d{4}\)", "", title).strip()

# Poster von TMDb abrufen
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

# Textuelle Erklärung generieren
def generate_text_explanation(movie_row, tags_selected):
    import random
    reasons = []

    genre_sim = movie_row.get("genre_similarity", 0)
    tag_sim = movie_row.get("tag_similarity", 0)
    rating = movie_row.get("avg_rating", 0)

    if genre_sim > 0.65:
        reasons.append("weil der Film starke Genre-Überschneidungen mit deinen Lieblingsfilmen hat")
    elif genre_sim > 0.4:
        reasons.append("wegen gewisser inhaltlicher Ähnlichkeiten in den Genres")

    if tag_sim > 0.4 and tags_selected:
        reasons.append("weil der Film viele deiner gewählten Tags abdeckt")
    elif tag_sim > 0.2 and tags_selected:
        reasons.append("weil der Film in Teilen zu deinen gewählten Tags passt")

    if rating >= 4.0:
        reasons.append("weil er von anderen Nutzer:innen besonders gut bewertet wurde")
    elif rating >= 3.6:
        reasons.append("weil der Film solide bewertet wurde")

    trust_score = movie_row.get("similarity", 0)
    trust_percent = round(trust_score * 100, 1)

    if trust_score >= 0.8:
        trust_label = "sehr hohen Vertrauen"
    elif trust_score >= 0.6:
        trust_label = "hohen Vertrauen"
    elif trust_score >= 0.4:
        trust_label = "mittlerem Vertrauen"
    else:
        trust_label = "niedrigem Vertrauen"

    vertrauen_text = f" Der Vertrauenswert dieser Empfehlung beträgt {trust_percent} %, was einem {trust_label} entspricht."

    if reasons:
        return "Dieser Film wurde empfohlen, " + " und ".join(reasons) + "." + vertrauen_text
    else:
        return "Dieser Film wurde empfohlen, weil er in mehreren Aspekten zu deinem Profil passt." + vertrauen_text


# === Google Drive CSV-Download mit Prüfung ===
def download_and_verify_csv(file_id, dest_path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    if not os.path.exists(dest_path):
        gdown.download(url, dest_path, quiet=False)

    with open(dest_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        if "<html" in first_line.lower():
            st.error(f"❌ Fehler beim Download: '{dest_path}' enthält HTML statt CSV.")
            st.stop()

# Verzeichnis vorbereiten
os.makedirs("./data", exist_ok=True)

# Downloads
download_and_verify_csv("1AVtktDFEXey1RSTq_lTFE4sgG-S9nIxT", "./data/movies.csv")
download_and_verify_csv("17USu4Dkt0SaoL8XiV3ckm1wX2iP7HgQQ", "./data/ratings.csv")
download_and_verify_csv("1wwWoz4RI9ysYVe5mtqNh7BBJ5JwL9IZj", "./data/genome-tags.csv")
download_and_verify_csv("1M0v8mSSbgS7Wz1HoMdCM_YqpXTh0bGd9", "./data/genome-scores.csv")

# === Daten laden ===
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

# === Recommender UI ===
st.title("🎬 Dein personalisierter Filmempfehler")
st.markdown("Dieses interaktive Empfehlungssystem schlägt dir Filme vor, die zu deinem Geschmack passen.")

st.markdown("Wähle dazu bitte 5 Filme aus, die dir besonders gut gefallen. Optional kannst du zusätzlich Tags wählen, um die Empfehlungen weiter zu verfeinern.")

st.markdown("Du kannst die Empfehlungen auf Filme ab einem bestimmten Jahr begrenzen (max. 2015):")
min_year = st.slider("Zeige Filme ab Jahr:", 1950, 2015, 1999)
movies = movies[movies["year"] >= min_year]

available_movies = movies.sort_values("title")
selected_titles = st.multiselect("Wähle 5 Filme:", available_movies["title"].tolist(), max_selections=5)

tags_selected = []
with st.expander("🔖 Optional: Wähle Tags, die dich interessieren"):
    all_tags = genome_tags["tag"].sort_values().unique().tolist()
    tags_selected = st.multiselect("Wähle bis zu 5 Tags:", all_tags, max_selections=5)

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

    st.subheader("🌟 Deine Filmempfehlungen")
    api_key = st.secrets["TMDB_API_KEY"]

    for _, row in top_movies.iterrows():
        col1, col2 = st.columns([1, 3])

        with col1:
            poster_url = get_movie_poster(clean_title(row["title"]), api_key)
            st.image(poster_url if poster_url else "https://via.placeholder.com/120x180.png?text=No+Image", width=300)

        with col2:
            st.markdown(f"<h4 style='margin-bottom:0.2em'>{row['title']}</h4>", unsafe_allow_html=True)
            st.markdown(" <h3>Textuelle Erklärung</h3>", unsafe_allow_html=True)
            explanation = generate_text_explanation(row, tags_selected)
            st.markdown(f"<i>{explanation}</i>", unsafe_allow_html=True)






