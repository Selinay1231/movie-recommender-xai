# MovieMate – Recommender mit Hero, Grid-Auswahl (ohne Lücken), Empfehlungen & abwechslungsreichen Texten

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
  color: #111 !important;
}

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
}
.hero__title {
  font-size: clamp(24px, 6vw, 44px);
  font-weight: 800;
  margin: 0 0 4px;
  color: #fff !important;
  text-shadow: 0 2px 6px rgba(0,0,0,.7);
}
.hero__subtitle {
  font-size: clamp(15px, 4vw, 20px);
  margin: 8px 0 0;
  color: #fff !important;
  opacity: .95;
  text-shadow: 0 1px 4px rgba(0,0,0,.6);
}

/* Cards */
.card {
  background: var(--card-bg); border-radius: 14px; overflow: hidden;
  box-shadow: 0 6px 14px rgba(0,0,0,.06); margin-bottom: 20px;
  display: flex; flex-direction: column; align-items: center;
}
.card img { width: 100%; height: 260px; object-fit: cover; border-bottom: 1px solid #eee; }
.card__title {
  font-size: 14px; font-weight: 700; color: #111 !important;
  height: 44px; display:flex; align-items:center; justify-content:center; text-align:center;
  overflow:hidden; text-overflow:ellipsis; margin:8px 0;
}
.card__explain { 
  font-size: 14px; 
  line-height: 1.6;       /* mehr Luft zwischen den Zeilen */
  text-align: justify;    /* schöner Blocksatz */
  margin-top: 10px;       /* Abstand zum Titel */
  padding: 0 6px;         /* links/rechts etwas Abstand geben */
  color: #111 !important;
}

.badge { 
  display: inline-block; 
  background: #eef2ff; 
  color: #4338ca; 
  padding: 4px 10px; 
  border-radius: 999px; 
  font-size: 12px; 
  font-weight: 700; 
  margin-bottom: 10px; 
}

.section-title { 
  margin: 16px 0 12px; 
  font-weight: 800; 
  color: #111 !important; 
}


/* Netflix Style Buttons */
.stButton > button {
  background-color: #e50914 !important;
  color: #fff !important;
  font-weight: 700 !important;
  border: none !important;
  border-radius: 6px !important;
  padding: 14px 28px !important;
  font-size: 16px !important;
  text-transform: uppercase !important;
  letter-spacing: .5px !important;
  box-shadow: 0 6px 20px rgba(229,9,20,.4) !important;
  transition: background .2s ease, transform .1s ease;
  width: 100% !important;
  margin-top: 20px !important; 
}
.stButton > button:hover {
  background-color: #f6121d !important;
  transform: scale(1.03) !important;
}
.stButton > button:disabled {
  opacity: .5 !important;
  cursor: not-allowed !important;
}
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
  
def fuzzy_filter(query, choices, limit=100, threshold=60):
    q = query.lower()

    # 1. Substring-Treffer (immer behalten)
    substring_matches = [c for c in choices if q in c.lower()]

    # 2. Fuzzy-Treffer
    results = process.extract(query, choices, limit=limit, score_cutoff=threshold)
    fuzzy_matches = [title for title, score, _ in results]

    # 3. Kombinieren: Substring-Treffer zuerst, Rest danach
    all_matches = substring_matches + [m for m in fuzzy_matches if m not in substring_matches]
    return all_matches



# =========================
# Poster laden (mit Caching)
# =========================
@st.cache_data(show_spinner=False)
def get_movie_poster(title, api_key):
    """Lädt ein Poster über TMDb und cached das Ergebnis."""
    if not api_key:
        return None
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


import openai

# OpenAI API-Key aus Streamlit Secrets
openai.api_key = st.secrets.get("OPENAI_API_KEY")
if not openai.api_key:
    st.error("❌ OPENAI_API_KEY fehlt in den Streamlit Secrets.")

def generate_text_explanation(movie_row, tags_selected):
    """
    Generiert eine textuelle Erklärung für die Filmempfehlung mit der OpenAI API.
    Nimmt Daten aus TMDb (falls vorhanden) und Similarity-Werte als Kontext.
    """

    title = movie_row.get("title", "Unbekannter Film")
    year = int(movie_row.get("year", 0)) if not pd.isna(movie_row.get("year", 0)) else None
    avg_rating = movie_row.get("avg_rating", 0)
    genres = str(movie_row.get("genres", ""))
    similarity = float(movie_row.get("similarity", 0))

    # TMDb API-Key holen, um Plot (overview) zu laden
    tmdb_key = st.secrets.get("TMDB_API_KEY")
    overview = ""
    if tmdb_key:
        try:
            url = "https://api.themoviedb.org/3/search/movie"
            params = {"api_key": tmdb_key, "query": title}
            r = requests.get(url, params=params, timeout=8)
            if r.ok and r.json().get("results"):
                overview = r.json()["results"][0].get("overview", "")
        except Exception:
            pass

    # Prompt für OpenAI
    prompt = f"""
    Erkläre in 2-3 Sätzen, warum der Film "{title}" empfohlen wird.
    Infos:
    - Jahr: {year}
    - Genres: {genres}
    - Durchschnittsbewertung: {avg_rating:.1f}
    - Plot: {overview}
    - Ähnlichkeitsscore: {similarity:.2f}

    Die Erklärung soll:
    - leicht verständlich und freundlich sein,
    - Bezug zu den Nutzerpräferenzen herstellen,
    - betonen, was ähnlich UND was neu/anders ist ("similar but different").
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"Dieser Film passt zu deinem Profil (Fehler bei Textgenerierung: {e})."


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
# ggf. IDs anpassen
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

    if {"movieId","rating"}.issubset(ratings.columns):
        agg = ratings.groupby("movieId")["rating"].agg(avg_rating="mean", n_ratings="count")
        movies = movies.merge(agg, on="movieId", how="left")
        movies["avg_rating"] = movies["avg_rating"].fillna(0)
        movies["n_ratings"] = movies["n_ratings"].fillna(0)
        # optionaler Qualitätsfilter
        movies = movies[(movies["avg_rating"] >= 3) & (movies["n_ratings"] >= 50)]

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

# ---------- MAIN ----------
else:
    min_year = st.slider("Zeige Filme ab Jahr:", 1950, 2015, 1999)

    # Tags direkt in der Auswahlphase anzeigen
    all_tags = genome_tags["tag"].astype(str).sort_values().unique().tolist()
    tags_selected = st.multiselect(
        "🔖 Tags auswählen (optional, max. 5)", 
        all_tags, 
        max_selections=5
    )

    # Auswahlphase: Grid anzeigen, solange <5 gewählt
    if len(st.session_state.selected_titles) < 5:
        search = st.text_input("🔎 Film suchen oder aus Liste wählen:")
        movies_view = movies[movies["year"] >= min_year].copy()
        available_movies = movies_view.sort_values("title")
        if search:
            # Alle Filme, die den eingegebenen Text irgendwo im Titel enthalten
            mask = available_movies["title"].str.contains(
                search, case=False, na=False, regex=False
            )
            available_movies = available_movies[mask].copy()
        
            # Filme, die direkt mit dem Text beginnen, nach oben sortieren
            available_movies["starts"] = available_movies["title"].str.lower().str.startswith(search.lower())
            available_movies = available_movies.sort_values(
                by=["starts","title"], ascending=[False, True]
            )
            available_movies = available_movies.drop(columns=["starts"])



        page_size = 25
        total_pages = max(1, (len(available_movies) - 1) // page_size + 1)
        start = st.session_state.search_page * page_size
        end = start + page_size
        page_movies = available_movies.iloc[start:end]

        # Zeilenweises Grid ohne Lücken (je 5 Karten pro Zeile)
        for i in range(0, len(page_movies), 5):
            cols = st.columns(5)
            for j, (_, row) in enumerate(page_movies.iloc[i:i+5].iterrows()):
                with cols[j]:
                    api_key = st.secrets.get("TMDB_API_KEY")
                    poster = get_movie_poster(clean_title(row["title"]), api_key) if api_key else None
                    poster = poster or "https://via.placeholder.com/300x450.png?text=No+Image"
                    st.markdown(f"""
                    <div class="card">
                      <img src="{poster}">
                      <div class="card__title">{row['title']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    is_selected = row["title"] in st.session_state.selected_titles
                    label = "✅ Entfernen" if is_selected else "➕ Auswählen"
                    if st.button(label, key=f"btn_{row['movieId']}"):
                        if is_selected:
                            st.session_state.selected_titles.remove(row["title"])
                        elif len(st.session_state.selected_titles) < 5:
                            st.session_state.selected_titles.append(row["title"])

        col1, col2, col3 = st.columns([1,2,1])
        with col1:
            if st.button("⬅️ Zurück", disabled=st.session_state.search_page == 0):
                st.session_state.search_page -= 1; st.rerun()
        with col3:
            if st.button("➡️ Weiter", disabled=st.session_state.search_page >= total_pages - 1):
                st.session_state.search_page += 1; st.rerun()

        st.progress(len(st.session_state.selected_titles)/5)
        st.write(f"Ausgewählt: {len(st.session_state.selected_titles)}/5 Filme")

    # Empfehlungsphase: Grid ausblenden, nur Empfehlungen zeigen
    else:
        st.success("✅ Du hast 5 Filme ausgewählt – hier deine Empfehlungen:")


        sel_key = selection_hash(st.session_state.selected_titles, tags_selected, int(min_year))
        if st.session_state.selection_key != sel_key:
            st.session_state.selection_key = sel_key
            st.session_state.rec_index = 3

        # IDs immer aus vollständigem Datensatz (nicht aus gefiltertem View)
        selected_ids = movies.loc[movies["title"].isin(st.session_state.selected_titles), "movieId"] \
                           .dropna().astype(int).values

        # Genre-Profil (Full) -> Vergleich (View)
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
        movies_view_rec["genre_similarity"] = cosine_similarity(
            user_profile.values.reshape(1, -1), view_genres.values
        )[0]

        # Tags (optional)
        movies_view_rec["tag_similarity"] = 0.0
        if tags_selected:
            tag_matrix = pd.pivot_table(
                genome_scores, values="relevance", index="movieId", columns="tagId", fill_value=0
            )
            selected_tag_ids = genome_tags[genome_tags["tag"].isin(tags_selected)]["tagId"] \
                                    .dropna().astype(int).tolist()
            user_tag_vector = pd.Series(0.0, index=tag_matrix.columns)
            for t in selected_tag_ids:
                if t in user_tag_vector.index: user_tag_vector.loc[t] = 1.0
            tag_block = tag_matrix.reindex(movies_view_rec["movieId"].values, fill_value=0).fillna(0)
            movies_view_rec["tag_similarity"] = cosine_similarity(
                user_tag_vector.values.reshape(1, -1), tag_block.values
            )[0]

        w_tag = 0.5 if tags_selected else 0.0
        w_genre = 1.0 - w_tag
        movies_view_rec["similarity"] = (
            w_genre*movies_view_rec["genre_similarity"] + w_tag*movies_view_rec["tag_similarity"]
        )

        sorted_movies = movies_view_rec.loc[~movies_view_rec["movieId"].isin(selected_ids)] \
                                       .sort_values("similarity", ascending=False) \
                                       .reset_index(drop=True)

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
                exp = generate_text_explanation(row, tags_selected)
                st.markdown(f"""
                <div class="card">
                  <img src="{poster}">
                  <div class="card__body">
                    <div class="badge">Empfehlung</div>
                    <div class="card__title">{row['title']}</div>
                    <div class="card__explain">{exp}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        can_more = show_n < max_n
        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            if st.button("🔄 Mehr Empfehlungen laden", disabled=not can_more, use_container_width=True):
                st.session_state.rec_index = min(st.session_state.rec_index + 3, max_n)
                st.rerun()




















