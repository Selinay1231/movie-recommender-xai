# MovieMate – eleganter Movie-Recommender (Hero Landing + Grid Cards + Fix Farben + weiße Inputs + abwechslungsreiche Texte)

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

/* Headline */
h1 { font-weight: 800; letter-spacing: .3px; color: #111 !important; }

/* Labels & Widget-Texte */
label, .stSelectbox label, .stMultiSelect label, .stSlider label { color: #111 !important; font-weight: 600; }
.stSlider p { color: #111 !important; }

/* Eingabefelder: Hintergrund immer weiß */
.stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"],
.stTextInput input {
  background: #fff !important;
  color: #111 !important;
  border: 1px solid #ccc !important;
  border-radius: 6px !important;
}

/* Buttons – Netflix Style */
div.stButton { 
  display:flex; 
  justify-content:center; 
}
div.stButton > button:first-child {
  background: #e50914;         /* Netflix Rot */
  color: #fff; 
  border: none; 
  border-radius: 4px;          /* fast eckig */
  padding: 16px 36px;          /* größer */
  font-size: 20px; 
  font-weight: 700; 
  text-transform: uppercase;   /* Netflix typisch */
  letter-spacing: .5px;
  box-shadow: 0 6px 20px rgba(229,9,20,.4);
  transition: background .2s ease, transform .1s ease;
}
div.stButton > button:first-child:hover {
  background: #f6121d;         /* helleres Rot beim Hover */
  transform: scale(1.03); 
}
div.stButton > button:first-child:disabled {
  opacity: .5; 
  cursor: not-allowed; 
}

/* Hero */
.hero {
  position: relative; border-radius: 18px; overflow: hidden;
  box-shadow: 0 10px 40px rgba(0,0,0,.08); margin-top: 8px;
}
.hero__bg {
  background-image: url('https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?auto=format&fit=crop&w=1600&q=80');
  background-size: cover; background-position: center; height: 290px;
}
.hero__content {
  position: absolute; inset: 0; display: flex; flex-direction: column;
  align-items: center; justify-content: center; text-align: center;
  padding: 0 24px; background: rgba(0,0,0,0.10);
}
.hero__title {
  font-size: clamp(24px, 6vw, 44px); font-weight: 800;
  margin: 0 0 4px; color: #fff !important;
}
.hero__subtitle {
  font-size: clamp(14px, 4vw, 18px); margin: 8px 0 0;
  color: #fff !important; opacity: .95;
}

/* Cards */
.grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
.card {
  background: var(--card-bg); border-radius: 14px; overflow: hidden;
  box-shadow: 0 8px 20px rgba(0,0,0,.06);
  transition: transform .08s ease, box-shadow .2s ease;
}
.card:hover { transform: translateY(-3px); box-shadow: 0 12px 28px rgba(0,0,0,.12); }
.card img {
  width: 100%; height: 300px; object-fit: cover;
  border-bottom: 1px solid #eee; background: #e5e7eb;
}
.card__body { padding: 14px 16px 18px; }
.card__title { margin: 0 0 8px; font-size: 17px; font-weight: 700; color: #111 !important; }
.card__explain { color: #374151 !important; line-height: 1.45; font-size: 15px; }
.badge {
  display: inline-block; background: #eef2ff; color: #4338ca;
  padding: 4px 10px; border-radius: 999px;
  font-size: 12px; font-weight: 700; margin-bottom: 8px;
}
.section-title { margin: 10px 0 8px; font-weight: 800; letter-spacing: .2px; color: #111 !important; }
</style>
"""), unsafe_allow_html=True)

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

    # --- Textbausteine ---
    genre_high = [
        "🎭 passt perfekt zu deinen Lieblingsgenres",
        "🎭 ist inhaltlich stark an deine bevorzugten Genres angelehnt",
        "🎭 spiegelt viele deiner Genre-Vorlieben wider"
    ]
    genre_mid = [
        "🎭 enthält einige Elemente deiner bevorzugten Genres",
        "🎭 überschneidet sich teilweise mit deinen Genre-Präferenzen",
        "🎭 bringt bekannte Genre-Themen mit"
    ]
    tag_texts = [
        "🔖 greift viele deiner gewählten Schlagworte auf",
        "🔖 deckt sich mit den von dir markierten Themen",
        "🔖 spiegelt deine Tag-Auswahl deutlich wider"
    ]
    rating_high = [
        "⭐ zählt zu den bestbewerteten Filmen seiner Art",
        "⭐ hat außergewöhnlich gute Bewertungen",
        "⭐ wird von vielen Zuschauer:innen als Highlight gesehen"
    ]
    rating_mid = [
        "⭐ wurde solide und überdurchschnittlich bewertet",
        "⭐ gilt als empfehlenswert in seiner Kategorie",
        "⭐ hat viele positive Stimmen erhalten"
    ]
    popular_texts = [
        "🎬 ist extrem beliebt und oft gesehen",
        "🎬 wurde schon tausendfach bewertet",
        "🎬 erfreut sich großer Bekanntheit"
    ]
    classic_texts = [
        "🕰 gilt als zeitloser Klassiker",
        "🕰 ist ein Film, der bis heute relevant geblieben ist",
        "🕰 wird seit Jahrzehnten geschätzt"
    ]
    modern_texts = [
        "✨ bringt moderne Themen auf die Leinwand",
        "✨ ist ein aktuellerer Film mit frischem Stil",
        "✨ greift zeitgemäße Inhalte auf"
    ]

    # --- Regeln ---
    if genre_sim > 0.65: reasons.append(random.choice(genre_high))
    elif genre_sim > 0.4: reasons.append(random.choice(genre_mid))
    if tag_sim > 0.4 and tags_selected: reasons.append(random.choice(tag_texts))
    if rating >= 4.0: reasons.append(random.choice(rating_high))
    elif rating >= 3.6: reasons.append(random.choice(rating_mid))
    if n_ratings >= 5000: reasons.append(random.choice(popular_texts))
    elif n_ratings >= 1000: reasons.append("🎬 hat viele Bewertungen gesammelt")
    if year and year > 2010: reasons.append(random.choice(modern_texts))
    elif year and year < 2000: reasons.append(random.choice(classic_texts))

    trust = movie_row.get("similarity", 0)
    trust_percent = round(trust * 100, 1)
    trust_label = "sehr hoch" if trust >= 0.8 else "hoch" if trust >= 0.6 else "mittel"
    vt = f"🔒 Vertrauenswert: {trust_percent}% ({trust_label})"

    if reasons:
        return "Dieser Film wurde empfohlen, " + " und ".join(reasons[:3]) + ". " + vt
    return "Dieser Film passt zu deinem Profil. " + vt

def download_and_verify_csv(file_id, dest_path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    if not os.path.exists(dest_path): gdown.download(url, dest_path, quiet=False)
    with open(dest_path, "r", encoding="utf-8") as f:
        if "<html" in f.readline().lower():
            st.error(f"❌ Fehler beim Download: '{dest_path}' enthält HTML statt CSV.")
            st.stop()

def selection_hash(titles, tags, year_from):
    raw = "|".join(sorted(titles)) + "||" + "|".join(sorted(tags)) + f"||{year_from}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

# =========================
# Daten laden
# =========================
os.makedirs("./data", exist_ok=True)
download_and_verify_csv("1AVtktDFEXey1RSTq_lTFE4sgG-S9nIxT","./data/movies.csv")
download_and_verify_csv("17USu4Dkt0SaoL8XiV3ckm1wX2iP7HgQQ","./data/ratings.csv")
download_and_verify_csv("1wwWoz4RI9ysYVe5mtqNh7BBJ5JwL9IZj","./data/genome-tags.csv")
download_and_verify_csv("1M0v8mSSbgS7Wz1HoMdCM_YqpXTh0bGd9","./data/genome-scores.csv")

@st.cache_data
def load_data():
    base = "./data/"
    movies = pd.read_csv(base+"movies.csv", sep=";", encoding="utf-8")
    ratings = pd.read_csv(base+"ratings.csv", sep=";", encoding="utf-8")
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)").astype(float)
    avg = ratings.groupby("movieId")["rating"].mean()
    cnt = ratings.groupby("movieId")["rating"].count()
    movies = movies.join(avg.rename("avg_rating"), on="movieId").join(cnt.rename("n_ratings"), on="movieId")
    movies = movies[(movies["avg_rating"] >= 3) & (movies["n_ratings"] >= 50)]
    return movies.reset_index(drop=True), ratings

@st.cache_data
def load_tag_data():
    base = "./data/"
    return (
        pd.read_csv(base+"genome-tags.csv", sep=";", encoding="utf-8"),
        pd.read_csv(base+"genome-scores.csv", sep=";", encoding="utf-8")
    )

movies, ratings = load_data()
genome_tags, genome_scores = load_tag_data()

# =========================
# UI
# =========================
st.markdown("<h1 style='text-align:center;'>🎬 MovieMate</h1>", unsafe_allow_html=True)

# ---------- HERO / INTRO ----------
if not st.session_state.intro_done:
    hero_html = dedent("""
    <div class="hero">
      <div class="hero__bg"></div>
      <div class="hero__content">
        <div class="hero__title">Willkommen bei MovieMate</div>
        <div class="hero__subtitle">Hier findest du Filme, die perfekt zu deinem Geschmack passen.</div>
        <div class="hero__subtitle" style="margin-top:6px;">Wähle dafür lediglich 5 Filme, die du magst.</div>
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

# ---------- MAIN RECOMMENDER ----------
else:
    st.markdown("<h3 class='section-title'>✨ Deine Auswahl</h3>", unsafe_allow_html=True)

    min_year = st.slider("Zeige Filme ab Jahr:", 1950, 2015, 1999)
    movies_view = movies[movies["year"] >= min_year].copy()
    available_movies = movies_view.sort_values("title")

    selected_titles = []
    for i in range(1,6):
        if i == 1 or len(selected_titles) >= (i - 1):
            film = st.selectbox(
                f"🎥 Film {i} auswählen oder suchen:",
                ["-- auswählen / suchen --"] + available_movies["title"].tolist(),
                key=f"film_{i}"
            )
            if film != "-- auswählen / suchen --":
                selected_titles.append(film)

    tags_selected = []
    if len(selected_titles) == 5:
        with st.expander("🔖 Optional: Tags auswählen"):
            all_tags = genome_tags["tag"].sort_values().unique().tolist()
            tags_selected = st.multiselect("Bis zu 5 Tags:", all_tags, max_selections=5)

    if len(selected_titles) == 5:
        sel_key = selection_hash(selected_titles, tags_selected, int(min_year))
        if st.session_state.selection_key != sel_key:
            st.session_state.selection_key = sel_key
            st.session_state.rec_index = 3

        selected_ids = movies_view[movies_view["title"].isin(selected_titles)]["movieId"].values
        movie_features = movies_view.join(movies_view["genres"].str.get_dummies("|"))
        genre_cols = movies_view["genres"].str.get_dummies("|").columns
        user_profile = movie_features[movie_features["movieId"].isin(selected_ids)][genre_cols].mean().values.reshape(1,-1)
        all_profiles = movie_features[genre_cols].values
        movies_view["genre_similarity"] = cosine_similarity(user_profile, all_profiles)[0]

        tag_matrix = pd.pivot_table(genome_scores, values="relevance", index="movieId", columns="tagId", fill_value=0)
        selected_tag_ids = genome_tags[genome_tags["tag"].isin(tags_selected)]["tagId"].tolist()
        user_tag_vector = pd.Series(0, index=tag_matrix.columns, dtype=float)
        for t in selected_tag_ids: user_tag_vector[t] = 1.0
        movies_view["tag_similarity"] = cosine_similarity(
            [user_tag_vector], tag_matrix.reindex(movies_view["movieId"].values, fill_value=0).fillna(0).values
        )[0]

        movies_view["similarity"] = (
            0.5*movies_view["genre_similarity"] + 0.5*movies_view["tag_similarity"]
            if tags_selected else movies_view["genre_similarity"]
        )
        sorted_movies = movies_view[~movies_view["movieId"].isin(selected_ids)].sort_values("similarity", ascending=False).reset_index(drop=True)

        max_n=len(sorted_movies); show_n=min(st.session_state.rec_index, max_n)
        to_show = sorted_movies.iloc[:show_n]

        st.markdown("<h3 class='section-title'>🌟 Deine Empfehlungen</h3>", unsafe_allow_html=True)
        api_key = st.secrets.get("TMDB_API_KEY")

        cards = ['<div class="grid">']
        for _, row in to_show.iterrows():
            poster = get_movie_poster(clean_title(row["title"]), api_key) if api_key else None
            poster = poster or "https://via.placeholder.com/500x750.png?text=No+Image"
            exp = generate_text_explanation(row, tags_selected)
            cards.append(dedent(f"""
            <div class="card">
              <img src="{poster}" alt="Poster">
              <div class="card__body">
                <div class="badge">Empfehlung</div>
                <div class="card__title">{row['title']}</div>
                <div class="card__explain">{exp}</div>
              </div>
            </div>
            """))
        cards.append("</div>")
        st.markdown("".join(cards), unsafe_allow_html=True)

        can_more = show_n < max_n
        st.write("")
        cc1,cc2,cc3 = st.columns([1,2,1])
        with cc2:
            if st.button("🔄 Mehr Empfehlungen laden", disabled=not can_more, use_container_width=True):
                st.session_state.rec_index = min(st.session_state.rec_index + 3, max_n)
                st.rerun()

        if not can_more:
            st.caption("🎉 Du hast alle passenden Empfehlungen gesehen. Ändere deine Auswahl, um neue Vorschläge zu bekommen.")





