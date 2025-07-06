# Projekt: Interaktives Movie-Recommender-System mit XAI-Vergleich (MovieLens 20M)

import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import shap
import matplotlib.pyplot as plt
import os
from datetime import datetime
import requests
import uuid
import re
from sklearn.decomposition import PCA
import gdown
import random 



#Für die SHAP Erklärung
def build_shap_features(movie_features, genre_columns, tag_matrix, selected_tag_ids, movies):
    X_genres = movie_features[genre_columns]
    tag_matrix_filtered = tag_matrix[selected_tag_ids] if selected_tag_ids else pd.DataFrame(index=movies["movieId"], columns=[])
    tag_matrix_filtered = tag_matrix_filtered.reindex(movies["movieId"].values, fill_value=0).fillna(0)
    tag_matrix_filtered.columns = [f"tag_{col}" for col in tag_matrix_filtered.columns]
    X_ratings = movies[["avg_rating", "n_ratings"]].reset_index(drop=True)
    X_all = pd.concat([X_genres.reset_index(drop=True), tag_matrix_filtered.reset_index(drop=True), X_ratings], axis=1)
    return X_all


# Zufällige, einmalige User-ID erzeugen (in Session gespeichert)
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

# === Banner / Einleitung für die Umfrage am Anfang===
st.markdown("""
    <div style='background-color:#f0f2f6; padding:20px; border-radius:10px; border: 1px solid #ccc;'>
        <h2 style='color:#1f77b4;'>🧠 Umfrage zur Künstlichen Intelligenz</h2>
        <p>Ich bin Studentin an der Universität Hildesheim und führe im Rahmen meiner Masterarbeit eine wissenschaftliche Untersuchung zur Wahrnehmung von Künstlicher Intelligenz (KI) durch. Ziel der Studie ist es herauszufinden, wie Menschen sich KI vorstellen, welche Erwartungen sie an Entscheidungen von KI-Systemen haben und welche Formen der Erklärung als verständlich und hilfreich empfunden werden.</p>
        <p>Die Studie besteht aus drei Teilen: einer kurzen Einstiegsumfrage, einem interaktiven Empfehlungssystem für Filme sowie einer abschließenden Befragung zur Verständlichkeit der Entscheidungserklärungen.</p>
        <p><b>Die Teilnahme dauert ca. 5–7 Minuten und erfolgt anonym.</b></p>
        <p><b>Wichtig:</b> Damit Ihre Angaben gespeichert werden, klicken Sie am Ende bitte auf den Button <i>„Antworten absenden“</i>.</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# === Umfrageformular ===
st.subheader("1. Allgemeine Angaben")
age_group = st.selectbox("Wie alt sind Sie?", [
    "Unter 18", "18–24", "25–34", "35–44", "45–54", "55–64", "65 oder älter"
])
contact_ki = st.radio("Hatten Sie bereits persönlichen Kontakt mit KI-Systemen (z.B. Siri. ChatGPT, Alexa)?",
                      ["Ja, regelmäßig", "Ja, gelegentlich", "Nein, noch nie"])

st.subheader("2. Mentale Vorstellungen")
assoziation = st.text_input("Welcher Begriff fällt Ihnen spontan ein, wenn Sie 'Künstliche Intelligenz' hören?")
geschlecht_vorstellung = st.radio("Wenn Sie an eine KI denken – stellen Sie sich diese eher weiblich, männlich oder neutral vor?",
                                    ["Weiblich", "Männlich", "Neutral", "Keine Vorstellung"])

st.subheader("3. Kontrolle & Vertrauen")
kontrolleinstellung = st.radio("Würden Sie einer KI erlauben, Entscheidungen allein zu treffen – oder möchten Sie immer die Kontrolle behalten?",
                                ["Nur mit vollständiger Erklärung und Zustimmung",
                                 "Entscheidung kann automatisiert erfolgen, wenn ich zustimmen kann",
                                 "Entscheidung darf allein von der KI getroffen werden",
                                 "Keine klare Meinung"])

wichtige_aspekte = st.multiselect(
    "Was ist Ihnen bei Empfehlungen durch ein System besonders wichtig?",
    [
        "Treffgenauigkeit (passt gut zu meinem Geschmack)",
        "Erklärbarkeit (ich verstehe, warum etwas empfohlen wird)",
        "Einfachheit / intuitive Bedienung",
        "Personalisierung (es passt zu mir als Person)",
        "Transparenz, wie Daten verwendet werden",
        "Keine der genannten"
    ]
)

vertrauensbereiche = st.multiselect(
    "In welchen Bereichen würden Sie Entscheidungen einer KI eher vertrauen?",
    options=[
        "Produktempfehlungen (z.B. Filme, Bücher)",
        "Medizinische Diagnosen",
        "Sprachassistenten und Smart-Home-Geräte (z.B. Alexa, Siri, Google Assistant)",
        "Intelligente Fahrzeugsysteme (z.B. autonomes Fahren, Assistenzsysteme)",
        "Finanzielle Beratung (z.B. Kreditwürdigkeit)",
        "Schule, Studium oder Online-Lernen (z.B. adaptive Lernplattformen)",
        "Kunst und Kreativität (z.B. KI-generierte Bilder oder Texte)",
        "Keine der genannten"
    ]
)

erklaerung_vertrauen = st.radio("Würden Sie der KI ein höheres Vertrauen schenken, wenn die Entscheidung transparent dargelegt wird?",
                                 ["Ja, auf jeden Fall", "Ja, aber nur in bestimmten Bereichen",
                                  "Nein, die Erklärung ändert nichts", "Ich weiß es nicht"])

beeinflussung = st.slider("Wie wichtig ist es Ihnen, die Entscheidungen eines KI-Systems selbst beeinflussen oder korrigieren zu können?",
                          1, 5, 3, format="%d")

# === Absenden und Daten merken ===
if st.button("Umfrage abschließen und starten"):
    st.session_state.umfrage_abgeschlossen = True
    st.session_state.umfrage_data = {
        "user_id": st.session_state.user_id,
        "timestamp": datetime.now().isoformat(),
        "age_group": age_group,
        "contact_ki": contact_ki,
        "assoziation": assoziation,
        "geschlecht_ki": geschlecht_vorstellung,
        "kontrolleinstellung": kontrolleinstellung,
        "wichtig": "; ".join(wichtige_aspekte),
        "vertrauen_bereich": "; ".join(vertrauensbereiche),
        "erklaerung_einfluss": erklaerung_vertrauen,
        "beeinflussung_wichtigkeit": beeinflussung
    }
    st.success("Vielen Dank! Du kannst jetzt deine personalisierte Empfehlung erhalten.")
    st.markdown("---")

if "umfrage_abgeschlossen" not in st.session_state:
    st.stop()

# Funktion zum Bereinigen des Titels
def clean_title(title):
    return re.sub(r"\s*\(\d{4}\)", "", title).strip()

def get_movie_poster(title, api_key):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": api_key,
        "query": title,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json().get("results")
        if results:
            poster_path = results[0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None


#Textuelle Erklärung
def generate_text_explanation(movie_row, tags_selected):
    reasons = []

    genre_sim = movie_row.get("genre_similarity", 0)
    tag_sim = movie_row.get("tag_similarity", 0)
    rating = movie_row.get("avg_rating", 0)

    if genre_sim > 0.65:
        reasons.append(random.choice([
            "weil der Film starke Genre-Überschneidungen mit deinen Lieblingsfilmen hat",
            "da er inhaltlich gut zu deinen bevorzugten Genres passt",
            "weil das Genre sehr häufig in deinen ausgewählten Filmen vorkommt"
        ]))
    elif genre_sim > 0.4:
        reasons.append(random.choice([
            "weil das Genre teilweise mit deinem Geschmack übereinstimmt",
            "wegen gewisser inhaltlicher Ähnlichkeiten in den Genres",
            "weil sich der Film thematisch in deinem Interessensbereich bewegt"
        ]))

    if tag_sim > 0.4 and tags_selected:
        reasons.append(random.choice([
            "weil der Film viele deiner gewählten Tags abdeckt",
            "weil die Inhalte zu deinen Themeninteressen passen",
            "da er starke Übereinstimmungen mit deinen Tags zeigt"
        ]))
    elif tag_sim > 0.2 and tags_selected:
        reasons.append(random.choice([
            "weil der Film in Teilen zu deinen gewählten Tags passt",
            "da einige Schlagworte mit deinen Interessen übereinstimmen",
            "weil es gewisse Ähnlichkeiten bei den gewählten Themen gibt"
        ]))

    if rating >= 4.0:
        reasons.append(random.choice([
            "weil er von anderen Nutzer:innen besonders gut bewertet wurde",
            "da er eine hohe durchschnittliche Bewertung hat",
            "weil er allgemein sehr beliebt ist"
        ]))
    elif rating >= 3.6:
        reasons.append(random.choice([
            "weil der Film solide bewertet wurde",
            "weil viele ihn für sehenswert halten",
            "da er eine gute, aber nicht überragende Bewertung erhalten hat"
        ]))

    if reasons:
        return "Dieser Film wurde empfohlen, " + " und ".join(reasons) + "."
    else:
        return "Dieser Film wurde empfohlen, weil er in mehreren Aspekten zu deinem Profil passt."

def download_from_gdrive(file_id, dest_path):
    if not os.path.exists(dest_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, dest_path, quiet=False)

# Lade Daten bei Bedarf aus Google Drive herunter
os.makedirs("./data", exist_ok=True)

download_from_gdrive("1AVtktDFEXey1RSTq_lTFE4sgG-S9nIxT", "./data/movies.csv")
download_from_gdrive("17USu4Dkt0SaoL8XiV3ckm1wX2iP7HgQQ", "./data/ratings.csv")
download_from_gdrive("1wwWoz4RI9ysYVe5mtqNh7BBJ5JwL9IZj", "./data/genome-tags.csv")
download_from_gdrive("1M0v8mSSbgS7Wz1HoMdCM_YqpXTh0bGd9", "./data/genome-scores.csv")


movies = pd.read_csv(base_path + "movies.csv", sep=";")
st.write("Spalten:", movies.columns.tolist())
st.write("Erste Zeilen:", movies.head())


@st.cache_data
def load_data():
    base_path = "./data/"
    movies = pd.read_csv(base_path + "movies.csv", sep=";")
    ratings = pd.read_csv(base_path + "ratings.csv", sep=";", encoding="utf-8")
    # Zeige die ersten 10 Zeilen der ratings-Daten an
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
    genome_tags = pd.read_csv(base_path + "genome-tags.csv", sep=";")
    genome_scores = pd.read_csv(base_path + "genome-scores.csv", sep=";")
    return genome_tags, genome_scores

movies, ratings = load_data()
genome_tags, genome_scores = load_tag_data()

st.title("\U0001F3AC Dein personalisierter Filmempfehler")
st.markdown("Dieses interaktive Empfehlungssystem schlägt dir Filme vor, die zu deinem Geschmack passen.")
st.markdown("Wähle dazu bitte 5 Filme aus, die dir besonders gut gefallen. Optional kannst du zusätzlich Tags (z. B. 'spannend', 'visuell beeindruckend') wählen, um die Empfehlungen weiter zu verfeinern.")

st.markdown("Du kannst die Empfehlungen auf Filme ab einem bestimmten Jahr begrenzen (max. 2015):")
min_year = st.slider("Zeige Filme ab Jahr:", 1950, 2015, 1999)
movies = movies[movies["year"] >= min_year]

available_movies = movies.sort_values("title")
selected_titles = st.multiselect("Wähle Filme:", available_movies["title"].tolist(), max_selections=5)

# Tags optional
tags_selected = []
with st.expander("\U0001F516 Optional: Wähle Tags, die dich interessieren"):
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

    #Berechnung
    movies["similarity"] = 0.5 * movies["genre_similarity"] + 0.5 * movies["tag_similarity"] if tags_selected else movies["genre_similarity"]
    top_movies = movies[~movies["movieId"].isin(selected_ids)].sort_values("similarity", ascending=False).head(2)


    #SHAP Erklärung
    from sklearn.linear_model import LinearRegression

    X_shap = build_shap_features(movie_features, genre_columns, tag_matrix, selected_tag_ids, movies)
    model = LinearRegression()
    model.fit(X_shap, movies["similarity"])

    explainer = shap.Explainer(model, X_shap)
    shap_values = explainer(X_shap)
    #Ende SHAP Erklärung



    st.subheader("\U0001F3AF Deine Filmempfehlungen")
    api_key = st.secrets["TMDB_API_KEY"]

    for i, (_, row) in enumerate(top_movies.iterrows()):
        col1, col2 = st.columns([1, 3])
        with col1:
            poster_url = get_movie_poster(clean_title(row["title"]), api_key)
            if poster_url:
                st.image(poster_url, width=300)
            else:
                st.image("https://via.placeholder.com/120x180.png?text=No+Image", width=300)

        with col2:
            st.markdown(f"<h4 style='margin-bottom:0.2em'>{row['title']}</h4>", unsafe_allow_html=True)
            st.markdown("🧠 <b>1. Textuelle Erklärung</b>", unsafe_allow_html=True)
            explanation = generate_text_explanation(row, tags_selected)
            st.markdown(f"<i>{explanation}</i>", unsafe_allow_html=True)

            st.markdown("🧠 <b>2. SHAP-Visualisierung</b>", unsafe_allow_html=True)
            fig, ax = plt.subplots()
            shap.plots.bar(shap_values[i], max_display=5, show=False)
            st.pyplot(fig)

            # 🧠 Vektorraum-Erklärung pro Film
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_shap.values)

            pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
            pca_df["movieId"] = movies["movieId"].values
            pca_df["selected"] = pca_df["movieId"].isin(selected_ids)
            pca_df["recommended"] = pca_df["movieId"] == row["movieId"]
            user_point = (
                pca_df[pca_df["selected"]]["PC1"].mean(),
                pca_df[pca_df["selected"]]["PC2"].mean()
            )

            st.markdown("🧠 <b>3. Vektorraum-Erklärung</b>", unsafe_allow_html=True)
            fig_pca, ax = plt.subplots()
            ax.scatter(pca_df["PC1"], pca_df["PC2"], color="lightgray", alpha=0.3, label="Andere Filme")
            ax.scatter(pca_df[pca_df["selected"]]["PC1"], pca_df[pca_df["selected"]]["PC2"], color="blue",
                       label="Ausgewählte Filme")
            ax.scatter(pca_df[pca_df["recommended"]]["PC1"], pca_df[pca_df["recommended"]]["PC2"], color="green",
                       label="Diese Empfehlung")
            ax.scatter(user_point[0], user_point[1], color="red", marker="x", s=100, label="Nutzerprofil")
            ax.set_title("Position der Empfehlung im Merkmalsraum")
            ax.legend()
            st.pyplot(fig_pca)

    st.subheader("\U0001F5E3️ Dein Feedback")
    rating = st.slider("Wie gut passen die Empfehlungen?", 1, 5, 3)
    understanding = st.radio("Welche Erklärung war für dich verständlicher?", ["Textuelle Erklärung", "SHAP", "Tabelle"])
    transparency = st.radio("Wie wichtig ist dir Transparenz in KI?", ["Unwichtig", "Wichtig", "Sehr wichtig"])

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import io

# === Google Drive Zugriff vorbereiten ===
SERVICE_ACCOUNT_JSON = 'google_service_account.json'
FILE_ID = '1gwLySMTgA_OHXQlW-yqlj8oVsM_bvawe'  # Deine Drive-Datei-ID für feedback.csv

creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_JSON,
    scopes=["https://www.googleapis.com/auth/drive"]
)
drive_service = build("drive", "v3", credentials=creds)

# === Hilfsfunktionen ===
def load_feedback_csv(file_id):
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    return pd.read_csv(fh, sep=';')

def append_and_upload_feedback(new_data: dict, file_id: str):
    df = load_feedback_csv(file_id)
    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    df.to_csv("temp_feedback.csv", sep=';', index=False, encoding='utf-8-sig')
    media = MediaFileUpload("temp_feedback.csv", mimetype="text/csv")
    drive_service.files().update(fileId=file_id, media_body=media).execute()

# === Streamlit: Antwort absenden ===
if st.button("Antworten absenden"):
    final_data = st.session_state.umfrage_data.copy()
    final_data.update({
        "selected_titles": ", ".join(selected_titles),
        "selected_tags": ", ".join(tags_selected),
        "recommendation_rating": rating,
        "understanding_choice": understanding,
        "transparency_importance": transparency
    })

    append_and_upload_feedback(final_data, FILE_ID)
    st.success("Vielen Dank für dein Feedback! Die Daten wurden erfolgreich gespeichert.")
