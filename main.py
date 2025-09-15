# MovieMate – eleganter Movie-Recommender (Searchbar + Grid-Auswahl + Pagination + stabile Empfehlungen)

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
@import url('https://fonts.googleapis.com/css2?family=Montserrat+Alternates:wght@700&display=swap');
h1 {
  font-family: 'Montserrat Alternates', sans-serif !important;
  font-weight: 800 !important;
  letter-spacing: 1px;
  color: #111 !important;
}

/* Inputs immer weiß */
.stTextInput input, .stSelectbox div[data-baseweb="select"], .stMultiSelect div[data-baseweb="select"]{
  background:#fff !important; color:#111 !important; border:1px solid #ccc !important; border-radius:6px !important;
}

/* Buttons – Netflix-Style */
div.stButton { display:flex; justify-content:center; }
div.stButton > button:first-child{
  background:#e50914; color:#fff; border:none; border-radius:6px;
  padding:12px 24px; font-size:16px; font-weight:700; text-transform:uppercase; letter-spacing:.5px;
  box-shadow:0 6px 20px rgba(229,9,20,.35); transition: background .2s ease, transform .06s ease;
}
div.stButton > button:first-child:hover{ background:#f6121d; transform:scale(1.03); }

/* Hero */
.hero{ position:relative; border-radius:18px; overflow:hidden; box-shadow:0 10px 40px rgba(0,0,0,.08); margin-top:8px; }
.hero__bg{ background-image:url('https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?auto=format&fit=crop&w=1600&q=80'); background-size:cover; background-position:center; height:290px; }
.hero__content{ position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; padding:0 24px; }
.hero__title{ font-size:clamp(24px,6vw,44px); font-weight:800; margin:0 0 4px; color:#fff !important; text-shadow:0 2px 6px rgba(0,0,0,.7); }
.hero__subtitle{ font-size:clamp(15px,4vw,20px); margin:8px 0 0; color:#fff !important; opacity:.95; text-shadow:0 1px 4px rgba(0,0,0,.6); }

/* Grids */
.grid-5{ display:grid; grid-template-columns: repeat(5, 1fr); gap:20px; }
.grid-3{ display:grid; grid-template-columns: repeat(3, 1fr); gap:20px; }

/* Karten */
.card{ background:var(--card-bg); border-radius:14px; overflow:hidden; box-shadow:0 8px 20px rgba(0,0,0,.06); transition:transform .08s ease, box-shadow .2s ease; }
.card:hover{ transform:translateY(-3px); box-shadow:0 12px 28px rgba(0,0,0,.12); }
.card img{ width:100%; height:260px; object-fit:cover; border-bottom:1px solid #eee; background:#e5e7eb; }
.card__body{ padding:12px; display:flex; flex-direction:column; align-items:center; gap:8px; }
.card__title{
  font-size:14px; font-weight:700; color:#111 !important;
  height:44px; /* Fix: immer gleich hoher Titel-Balken */
  display:flex; align-items:center; justify-content:center; text-align:center;
  overflow:hidden; text-overflow:ellipsis;
}
.card__explain{ color:#374151 !important; line-height:1.45; font-size:15px; }
.badge{ display:inline-block; background:#eef2ff; color:#4338ca; padding:4px 10px; border-radius:999px; font-size:12px; font-weight:700; }
.section-title{ margin:10px 0 8px; font-weight:800; letter-spacing:.2px; color:#111 !important; }
.small-btn{ width:100%; }
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
def clean_title(title: str) -> str:
    return re.sub(r"\s*\(\d{4}\)", "", str(title)).strip()

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
    except Exception:
        pass
    return None

def generate_text_explanation(row, tags_selected):
    bits = [f"Wir empfehlen **{row['title']}** wegen hoher Genre-Ähnlichkeit."]
    if tags_selected:
        bits.append("Deine gewählten Tags passen zusätzlich gut.")
    sim = float(row.get("similarity", 0))
    bits.append(f"🔒 Vertrauenswert: {round(sim*100,1)}%.")
    return " ".join(bits)

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
# Daten laden (robust)
# =========================
def _read_csv_anysep(path):
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8")

os.makedirs("./data", exist_ok=True)
# Deine vorhandenen File-IDs – ggf. anpassen
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

    # year-Spalte sicherstellen
    if "year" not in movies.columns:
        movies["year"] = movies["title"].astype(str).str.extract(r"\((\d{4})\)")
    movies["year"] = pd.to_numeric(movies["year"], errors="coerce").fillna(0).astype(int)

    # Qualitätsmetriken (optional, aber nützlich)
    if {"movieId","rating"}.issubset(ratings.columns):
        agg = ratings.groupby("movieId")["rating"].agg(avg_rating="mean", n_ratings="count")
        movies = movies.merge(agg, on="movieId", how="left")
        movies["avg_rating"] = movies["avg_rating"].fillna(0)
        movies["n_ratings"] = movies["n_ratings"].fillna(0)
        # leicht filtern
        movies = movies[(movies["avg_rating"] >= 3) & (movies["n_ratings"] >= 50)]

    return movies.reset_index(drop=True), ratings, genome_tags, genome_scores

movies, ratings, genome_tags, genome_scores = load_data()

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
        <div class="hero__subtitle">Wähle 5 Filme über die Suche aus und erhalte sofort passende Empfehlungen.</div>
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

    # Jahr-Filter wirkt nur auf die durchblätterbare Liste (nicht auf already selected titles)
    min_year = st.slider("Zeige Filme ab Jahr:", 1950, 2015, 1999)

    # Suchfeld
    search = st.text_input("🔎 Film suchen oder aus Liste wählen:")

    # Daten für die Auswahl
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

    # Grid-Auswahl (Poster -> fester Titelbalken -> Button)
    grid_cards = ['<div class="grid-5">']
    # Wir rendern den Button separat (unterhalb des Titels), damit der Titelbalken fix ist.
    cols = st.columns(5)
    for idx, row in page_movies.iterrows():
        col = cols[idx % 5]
        with col:
            api_key = st.secrets.get("TMDB_API_KEY")
            poster = get_movie_poster(clean_title(row["title"]), api_key) if api_key else None
            poster = poster or "https://via.placeholder.com/300x450.png?text=No+Image"

            # Karte (Bild + fester Titel)
            st.markdown(dedent(f"""
            <div class="card">
              <img src="{poster}" alt="Poster">
              <div class="card__body">
                <div class="card__title">{row['title']}</div>
              </div>
            </div>
            """), unsafe_allow_html=True)

            is_selected = row["title"] in st.session_state.selected_titles
            label = "✅ Entfernen" if is_selected else "➕ Auswählen"
            if st.button(label, key=f"sel_{row['movieId']}", use_container_width=True):
                if is_selected:
                    st.session_state.selected_titles.remove(row["title"])
                elif len(st.session_state.selected_titles) < 5:
                    st.session_state.selected_titles.append(row["title"])
                # bei Änderung nicht sofort rerun, damit die Seite nicht „springt“

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

    # -----------------------------
    # Empfehlungen (aktiv nach 5 Filmen)
    # -----------------------------
    tags_selected = []
    if len(st.session_state.selected_titles) == 5:
        with st.expander("🔖 Optional: Tags auswählen"):
            all_tags = genome_tags["tag"].astype(str).sort_values().unique().tolist()
            tags_selected = st.multiselect("Bis zu 5 Tags:", all_tags, max_selections=5)

        # Schlüssel für Recompute erkennen
        sel_key = selection_hash(st.session_state.selected_titles, tags_selected, int(min_year))
        if st.session_state.selection_key != sel_key:
            st.session_state.selection_key = sel_key
            st.session_state.rec_index = 3  # zurücksetzen

        # IDs IMMER aus vollständigem movies (nicht aus dem gefilterten View)
        selected_ids = movies.loc[movies["title"].isin(st.session_state.selected_titles), "movieId"] \
                           .dropna().astype(int).values

        # --- Genre-Profil über FULL berechnen und dann mit View vergleichen ---
        genres_full = movies["genres"].astype(str).str.get_dummies("|")
        movie_features_full = movies.join(genres_full)

        user_rows = movie_features_full[movie_features_full["movieId"].isin(selected_ids)]
        genre_cols = genres_full.columns
        user_profile = user_rows[genre_cols].mean(axis=0)

        if user_profile.isna().all():
            st.warning("Deine Auswahl ergab kein verwertbares Genre-Profil. Bitte wähle andere Filme oder passe das Jahr an.")
            st.stop()

        # View (nach min_year gefiltert) auf gleiche Genre-Spalten bringen
        movies_view_rec = movies[movies["year"] >= min_year].copy()
        view_genres = movies_view_rec["genres"].astype(str).str.get_dummies("|")
        for c in genre_cols:
            if c not in view_genres.columns:
                view_genres[c] = 0
        view_genres = view_genres[genre_cols]

        # Genre-Similarity
        try:
            movies_view_rec["genre_similarity"] = cosine_similarity(
                user_profile.values.reshape(1, -1),
                view_genres.values
            )[0]
        except Exception:
            movies_view_rec["genre_similarity"] = 0.0

        # Tag-Similarity (optional)
        movies_view_rec["tag_similarity"] = 0.0
        if tags_selected:
            tag_matrix = pd.pivot_table(
                genome_scores, values="relevance", index="movieId", columns="tagId", fill_value=0
            )
            selected_tag_ids = genome_tags[genome_tags["tag"].isin(tags_selected)]["tagId"] \
                                    .dropna().astype(int).tolist()
            user_tag_vector = pd.Series(0.0, index=tag_matrix.columns)
            for t in selected_tag_ids:
                if t in user_tag_vector.index:
                    user_tag_vector.loc[t] = 1.0

            tag_block = tag_matrix.reindex(movies_view_rec["movieId"].values, fill_value=0).fillna(0)
            try:
                movies_view_rec["tag_similarity"] = cosine_similarity(
                    user_tag_vector.values.reshape(1, -1),
                    tag_block.values
                )[0]
            except Exception:
                movies_view_rec["tag_similarity"] = 0.0

        # Gesamtscore
        w_tag = 0.5 if tags_selected else 0.0
        w_genre = 1.0 - w_tag
        movies_view_rec["similarity"] = w_genre*movies_view_rec["genre_similarity"] + w_tag*movies_view_rec["tag_similarity"]

        # Ausgewählte ausschließen
        sorted_movies = (
            movies_view_rec.loc[~movies_view_rec["movieId"].isin(selected_ids)]
            .sort_values("similarity", ascending=False)
            .reset_index(drop=True)
        )

        max_n = len(sorted_movies)
        show_n = min(st.session_state.rec_index, max_n)
        to_show = sorted_movies.iloc[:show_n]

        st.markdown("<h3 class='section-title'>🌟 Deine Empfehlungen</h3>", unsafe_allow_html=True)

        # Empfehlungs-Karten (3 Spalten)
        rows = []
        rows.append('<div class="grid-3">')
        api_key = st.secrets.get("TMDB_API_KEY")
        for _, r in to_show.iterrows():
            poster = get_movie_poster(clean_title(r["title"]), api_key) if api_key else None
            poster = poster or "https://via.placeholder.com/500x750.png?text=No+Image"
            exp = generate_text_explanation(r, tags_selected)
            rows.append(dedent(f"""
            <div class="card">
              <img src="{poster}" alt="Poster">
              <div class="card__body">
                <div class="badge">Empfehlung</div>
                <div class="card__title">{r['title']}</div>
                <div class="card__explain">{exp}</div>
              </div>
            </div>
            """))
        rows.append("</div>")
        st.markdown("".join(rows), unsafe_allow_html=True)

        # Mehr laden
        can_more = show_n < max_n
        st.write("")
        c1,c2,c3 = st.columns([1,2,1])
        with c2:
            if st.button("🔄 Mehr Empfehlungen laden", disabled=not can_more, use_container_width=True):
                st.session_state.rec_index = min(st.session_state.rec_index + 3, max_n)
                st.rerun()

        if not can_more and max_n > 0:
            st.caption("🎉 Du hast alle passenden Empfehlungen gesehen. Ändere deine Auswahl oder Tags für neue Vorschläge.")

