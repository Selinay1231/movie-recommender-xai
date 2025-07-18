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
import time


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

st.subheader("1. Allgemeine Angaben")

# 1.1 Alter
age_group = st.selectbox("**1.1 Wie alt sind Sie?**", [
    "Unter 18", "18–24", "25–34", "35–44", "45–54", "55–64", "65 oder älter"
])



# 1.2 Kontakt mit KI
contact_ki = st.radio("**1.2 Hatten Sie bereits persönlichen Kontakt mit KI-Systemen (z. B. Siri, ChatGPT, Alexa)?**",
                      ["Ja, regelmäßig", "Ja, gelegentlich", "Nein, noch nie"])


# 1.3 Nutzungshäufigkeit (nur anzeigen, wenn vorheriger Kontakt)
if contact_ki != "Nein, noch nie":
    ki_frequency = st.radio("**1.3 Wie häufig nutzen Sie KI-Systeme?**",
                             ["Nie", "Wöchentlich", "Monatlich", "Täglich"])

# 1.4 Genutztes System
most_used_ki = st.selectbox("**1.4 Welches KI-System nutzen Sie am häufigsten?**", [
    "ChatGPT (OpenAI)", "Claude (Anthropic)", "Google Gemini", "Microsoft Copilot", "Perplexity AI"
])

# === 1.5 Wichtigkeit verschiedener Eigenschaften ===
st.markdown("**1.5 Was ist Ihnen besonders wichtig bei einem KI-System?**")
st.markdown(
    "<div style='margin-bottom: 12px;'>"
    "<b>Skala:</b> 1 = unwichtig  3 = neutral  5 = sehr wichtig"
    "</div>",
    unsafe_allow_html=True
)

treffgenauigkeit = st.slider("Treffgenauigkeit (passt gut zu meinem Geschmack)", 1, 5, 3)
transparenz = st.slider("Transparenz & Erklärbarkeit (ich verstehe, warum etwas empfohlen wird)", 1, 5, 3)
einfachheit = st.slider("Einfachheit / intuitive Bedienung", 1, 5, 3)
zugänglichkeit = st.slider("Zugänglichkeit", 1, 5, 3)
personalisierung = st.slider("Personalisierung (es passt zu mir als Person)", 1, 5, 3)
datenschutz = st.slider("Datenschutz", 1, 5, 3)


#################################################################################

# 1.6 KI-Aktivitäten (Mehrfachauswahl)
st.markdown("**1.6 Für welche der folgenden Aktivitäten verwenden Sie KI?**")
ki_aktivitaeten = st.multiselect(
    "Mehrfachauswahl möglich",
    [
        "Informationsbeschaffung (anstatt zu googlen)",
        "Ratschläge",
        "Lösung von technischen Problemen",
        "Vereinfachung von komplexen Inhalten",
        "Schreiben von Texten",
        "Übersetzen von Texten",
        "Ideenfindung / Brainstorming",
        "Planung & Strukturierung",
        "Smalltalk / Unterhaltung"
    ]
)


st.subheader("2. Mentale Vorstellungen")

# 2.1 Freitext: Erklärung für ein Kind
erklaerung_kind = st.text_area("**2.1 Wie würden Sie einem Kind erklären, was eine KI ist?**")

# 2.2 Freitext: Name für KI
ki_name = st.text_input("**2.2 Stellen Sie sich vor, Sie sprechen mit einer KI über ein persönliches Problem. Welchen Namen würden Sie dieser KI geben?**")

# 2.3 Vorstellungen über KI-Denken
ki_verstaendnis = st.radio("**2.3 Eine KI …**", [
    "denkt ähnlich wie ein Mensch",
    "verarbeitet nur Informationen, ohne zu denken",
    "imitiert Denken, versteht aber nichts",
    "keins davon"
])

# 2.4 Wahrnehmung der KI
ki_rollenbild = st.radio("**2.4 Ein KI-System ist für mich …**", [
    "Wie ein Werkzeug – es macht nur, was ich eingebe",
    "Wie ein Assistent – es hilft, aber trifft keine eigenen Entscheidungen",
    "Wie ein Agent – es handelt eigenständig",
    "keins davon"
])

# 2.5 Mensch vs. KI Aufgabenvergleich (Mehrfachauswahl möglich)
st.markdown("**2.5 Wer würde die Aufgabe besser lösen (Mensch vs. KI)?**")
aufgabenvergleich = {
    "Emotionen erkennen": st.radio("Emotionen erkennen", ["Mensch", "KI"]),
    "Menschen täuschend echt imitieren": st.radio("Menschen täuschend echt imitieren", ["Mensch", "KI"]),
    "Kreativ sein": st.radio("Kreativ sein", ["Mensch", "KI"]),
    "Moralisch handeln": st.radio("Moralisch handeln", ["Mensch", "KI"]),
    "Verantwortung übernehmen": st.radio("Verantwortung übernehmen", ["Mensch", "KI"]),
    "Selbst lernen ohne menschliche Hilfe": st.radio("Selbst lernen, ohne menschliche Hilfe", ["Mensch", "KI"]),
    "Keines dieser Dinge": st.radio("Keines dieser Dinge", ["Mensch", "KI"])
}

st.subheader("3. Kontrolle & Vertrauen")

# 3.1 Szenario Navigationssystem
navigation_entscheidung = st.radio(
    "**3.1 Ihr Navigationssystem erkennt plötzlich einen Stau und möchte automatisch eine neue Route wählen.**",
    ["Mach das sofort", "Zeig mir erst die Optionen", "Frag mich vorher", "Ich entscheide das lieber selbst"]
)

st.markdown("**3.3 Wem würden Sie in den folgenden Bereichen eher vertrauen?**")
st.markdown("_Skala: 0 = Mensch  10 = KI_")

vertrauen_produkte = st.slider("Produktempfehlungen (z. B. Filme, Bücher)", 0, 10, 5)
vertrauen_medizin = st.slider("Medizinische Diagnosen", 0, 10, 5)
vertrauen_verkehr = st.slider("Im Straßenverkehr (z. B. Navigation)", 0, 10, 5)
vertrauen_finanz = st.slider("Finanzielle Beratung (z. B. Kreditwürdigkeit)", 0, 10, 5)
vertrauen_bildung = st.slider("Schule und Studium", 0, 10, 5)
vertrauen_kunst = st.slider("Kunst und Kreativität", 0, 10, 5)


# 3.4 Transparenzfrage (direkt)
transparenz_vertrauen = st.radio(
    "**3.4 Würden Sie der KI ein höheres Vertrauen schenken, wenn die Entscheidung transparent dargelegt wird?**",
    ["Ja, auf jeden Fall", "Ja, aber nur in bestimmten Bereichen", "Nein, die Erklärung ändert nichts", "Ich weiß es nicht"]
)

# 3.4 Szenario Online-Jobplattform
job_szenario = st.radio(
    "**3.5 Sie nutzen eine Online-Plattform, die mithilfe von KI passende Jobangebote für Sie auswählt. Die Plattform schlägt Ihnen eine konkrete Stelle vor. Wie möchten Sie diese Empfehlung dargestellt bekommen?**",
    [
        "Die Stelle reicht mir, ich vertraue der Auswahl der KI.",
        "Ich möchte zusätzlich eine kurze Erklärung erhalten, z. B.: 'Diese Stelle wurde empfohlen, weil sie gut zu Ihren bisherigen Berufserfahrungen und Interessen passt.'",
        "Kommt auf die Art des Jobs oder den Kontext an",
        "Ich bin mir nicht sicher"
    ]
)

# 3.6 App-Einstellungen (Kontrollverhalten)
app_einstellungen = st.multiselect(
    "**3.6 Welche Einstellungen würden Sie bei einer neuen KI-basierten App am ehesten anpassen?**",
    [
        "Benachrichtigungen anpassen",
        "Empfehlungsalgorithmus konfigurieren",
        "Datenschutzeinstellungen überprüfen",
        "Automatische Funktionen deaktivieren"
    ]
)

# === 4. Verständnis von KI-Entscheidungen und Nutzung ===
st.subheader("**4. Verständnis von KI-Entscheidungen und Nutzung**")

ki_entscheidung = st.radio(
    "**Wie glauben Sie, trifft eine KI ihre Entscheidungen?**",
    [
        "Sie folgt Regeln, die Menschen ihr beigebracht haben",
        "Sie lernt selbstständig aus vielen Beispielen",
        "Sie ist einfach sehr schlau – sie weiß es irgendwie",
        "Ich weiß es nicht"
    ]
)

ki_unfaehigkeit = st.radio(
    "**Wenn eine KI eine Aufgabe nicht lösen kann – woran liegt das Ihrer Meinung nach?**",
    [
        "Sie kennt zu wenig Beispiele aus der Vergangenheit",
        "Sie versteht keine Regeln für diese Aufgabe",
        "Sie kann nicht so leicht adaptieren / ihr fehlt die nötige Logik",
        "Ich weiß es nicht"
    ]
)


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

        # Abschnitt 3: Kontrolle & Vertrauen
        "navigation_entscheidung": navigation_entscheidung,
        "vertrauen_produkte": vertrauen_produkte,
        "vertrauen_medizin": vertrauen_medizin,
        "vertrauen_verkehr": vertrauen_verkehr,
        "vertrauen_finanz": vertrauen_finanz,
        "vertrauen_bildung": vertrauen_bildung,
        "vertrauen_kunst": vertrauen_kunst,
        "transparenz_vertrauen": transparenz_vertrauen,
        "job_szenario": job_szenario,
        "app_einstellungen": "; ".join(app_einstellungen),

        # Abschnitt 4: Verständnis von KI
        "ki_entscheidung": ki_entscheidung,
        "ki_unfaehigkeit": ki_unfaehigkeit
    }
    st.success("Vielen Dank! Du kannst jetzt deine personalisierte Empfehlung erhalten.")
    st.markdown("---")




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

    # Vertrauenswert basierend auf Ähnlichkeit
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

    vertrauen_text = f" Der Vertrauenswert dieser Empfehlung beträgt {trust_percent} %, was einem {trust_label} entspricht."

    if reasons:
        return "Dieser Film wurde empfohlen, " + " und ".join(reasons) + "." + vertrauen_text
    else:
        return "Dieser Film wurde empfohlen, weil er in mehreren Aspekten zu deinem Profil passt." + vertrauen_text


import shutil

# CACHE Löschen falls es irgendwie nicht mehr geht ..Löscht den gesamten ./data Ordner, wenn er existiert
#if os.path.exists("./data"):
    #shutil.rmtree("./data")


def download_and_verify_csv(file_id, dest_path):
    # Direktlink zum Herunterladen
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    if not os.path.exists(dest_path):
        gdown.download(url, dest_path, quiet=False)

    # Dateiinhalt kurz prüfen
    with open(dest_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        if "<html" in first_line.lower():
            st.error(f"❌ Fehler beim Download: '{dest_path}' enthält HTML statt CSV. Vermutlich wurde ein Redirect oder Google-Warnung geladen.")
            st.stop()

# Verzeichnis vorbereiten
os.makedirs("./data", exist_ok=True)

# Downloads mit Überprüfung
download_and_verify_csv("1AVtktDFEXey1RSTq_lTFE4sgG-S9nIxT", "./data/movies.csv")
download_and_verify_csv("17USu4Dkt0SaoL8XiV3ckm1wX2iP7HgQQ", "./data/ratings.csv")
download_and_verify_csv("1wwWoz4RI9ysYVe5mtqNh7BBJ5JwL9IZj", "./data/genome-tags.csv")
download_and_verify_csv("1M0v8mSSbgS7Wz1HoMdCM_YqpXTh0bGd9", "./data/genome-scores.csv")

@st.cache_data
def load_data():
    base_path = "./data/"
    movies = pd.read_csv(base_path + "movies.csv", sep=";", encoding="utf-8")
    ratings = pd.read_csv(base_path + "ratings.csv", sep=";", encoding="utf-8")


    # Weiterverarbeitung
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
    top_movies = movies[~movies["movieId"].isin(selected_ids)].sort_values("similarity", ascending=False).head(3)


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

    # === Nachbefragung: Bewertung der Erklärformate ===
st.subheader("🗣️ Dein Feedback")

# Bewertungsskala mit erklärenden Labels
st.markdown("**Wie gut passen die Empfehlungen zu deinem Geschmack?**")
rating = st.slider("Skala: 1 = gar nicht passend, 3 = mittelmäßig, 5 = sehr passend", 1, 5, 3)

# Verständlichstes Erklärformat auswählen
understanding = st.radio(
    "**Welche Erklärung war für dich am verständlichsten?**",
    ["Vektorraumerklärung (Tabelle)", "SHAP-Erklärung", "Textuelle Erklärung"]
)

st.markdown("**Hat die Erklärung dein Vertrauen in die KI-Empfehlung gestärkt?**")
trust_effect = st.slider("Skala: 1 = gar nicht, 3 = neutral, 5 = sehr stark", 1, 5, 3)



# === Feedback Speicherung via Google Sheets ===
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import streamlit as st

# Verbindung zu Google Sheets via Streamlit Secrets (kein JSON-File nötig)
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["google_service_account"], scope)
client = gspread.authorize(creds)
sheet = client.open("KI_Umfrage_Responses").sheet1  # Muss existieren!

# === Button zum Absenden ===
if st.button("Antworten absenden"):
    try:
        umfrage_data = st.session_state.umfrage_data  # Vorher gespeichert

        # Werte extrahieren
        row = [
            umfrage_data["user_id"],
            umfrage_data["timestamp"],
            umfrage_data["age_group"],
            umfrage_data["contact_ki"],
            umfrage_data["assoziation"],
            umfrage_data["geschlecht_ki"],
            umfrage_data["kontrolleinstellung"],
            umfrage_data["wichtig"],
            umfrage_data["vertrauen_bereich"],
            umfrage_data["erklaerung_einfluss"],
            umfrage_data["beeinflussung_wichtigkeit"]
        ]

        # Header setzen (einmalig, wenn leer)
        if not sheet.get_all_values():  # Leeres Sheet
            header = [
                "user_id", "timestamp", "age_group", "contact_ki", "assoziation",
                "geschlecht_ki", "kontrolleinstellung", "wichtig",
                "vertrauen_bereich", "erklaerung_einfluss", "beeinflussung_wichtigkeit"
            ]
            sheet.append_row(header)

        # In nächste freie Zeile einfügen
        sheet.append_row(row)
        st.success("✅ Vielen Dank für deine Teilnahme! Deine Antworten wurden gespeichert.")
    except Exception as e:
        st.error(f"❌ Fehler beim Speichern der Antworten: {e}")
