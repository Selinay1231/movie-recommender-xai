# MovieMate – eleganter Movie-Recommender (Hero + Netflix-Style Button + 5er-Auswahlgrid + 3er-Empfehlungsgrid)

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

/* Karten (Auswahl & Empfehlungen) */
.card {
  background: var(--card-bg); border-radius: 14px; overflow: hidden;
  box-shadow: 0 6px 14px rgba(0,0,0,.06); margin-bottom: 20px;
  display: flex; flex-direction: column; align-items: center;
}
.card img { width: 100%; height: 260px; object-fit: cover; border-bottom: 1px solid #eee; background:#e5e7eb; }
.card__title {
  font-size: 14px; font-weight: 700; color: #111 !important;
  height: 44px; display:flex; align-items:center; justify-content:center; text-align:center;
  overflow:hidden; text-overflow:ellipsis; margin:8px 0;
}
.card__body { padding: 12px; display:flex; flex-direction:column; align-items:center; gap: 8px; }
.card__explain { font-size: 15px; line-height: 1.4; text-align: left; }
.badge { display:inline-block; background:#eef2ff; color:#4338ca; padding:4px 10px; border-radius:999px; font-size:12px; font-weight:700; margin-bottom:6px; }
.section-title { margin:12px 0 10px; font-weight:800; color:#111 !important; }

/* Grid für Empfehlungen (3 Spalten, fester Abstand) */
.grid { 
  display: grid; 
  grid-template-columns: repeat(3, 1fr); 
  gap: 20px; 
}

/* Netflix Style Buttons (gilt global inklusive Landingpage-Button) */
.stButton > button {
  background-color: #e50914 !important;
  color: #fff !important;
  font-weight: 700 !important;
  border: none !important;
  border-radius: 6px !important;
  padding: 14px 24px !important;
  font-size: 16px !important;
  text-transform: uppercase !important;
  letter-spacing: .5px !important;
  box-shadow: 0 6px 20px rgba(229,9,20,.4) !important;
  transition: background .2s ease, transform .1s ease;
  width: 100% !important;
}
.stButton > button:hover {
  background-color: #f6121d !important;
  transform: scale(1.03) !important;
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
    return re.sub(r"\\s*\\(\\d{4}\\)", "", str(title)).strip()

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

def generate_text_explanation(movie_row, tags_selected):
    # abwechslungsreiche Textbausteine (dein Ansatz)
    reasons = []
    genre_sim = movie_row.get("genre_similarity", 0)
    tag_sim = movie_row.get("tag_similarity", 0)
    rating = movie_row.get("avg_rating", 0)
    year = int(movie_row.get("year", 0)) if not pd.isna(movie_row.get("year", 0)) else None
    n_ratings = movie_row.get("n_ratings", 0)

    genre_high = [
        "weil er perfekt zu deinen Lieblingsgenres passt",
        "weil er inhaltlich stark an deine Genre-Vorlieben angelehnt ist",
        "denn er spiegelt viele deiner Genre-Präferenzen wider"
    ]
    genre_mid = [
        "weil er einige Elemente deiner Genres enthält",
        "denn er überschneidet sich teilweise mit deinen Vorlieben",
        "weil er bekannte Genre-Themen aufgreift"
    ]
    tag_texts = [
        "denn er deckt sich mit deinen gewählten Themen",
        "weil er deine Tag-Auswahl widerspiegelt",
        "denn er greift viele deiner Schlagworte auf"
    ]
    rating_high = [
        "denn er zählt zu den bestbewerteten Filmen",
        "denn er hat außergewöhnlich gute Bewertungen",
        "denn er wird von vielen Zuschauenden als Highlight gesehen"
    ]
    rating_mid = [
        "denn er wurde solide und überdurchschnittlich bewertet",
        "denn er gilt als empfehlenswert in seiner Kategorie",
        "denn er hat viele positive Stimmen erhalten"
    ]
    popular_texts = [
        "denn er ist extrem beliebt und wurde oft gesehen",
        "denn er wurde schon tausendfach bewertet",
        "denn er erfreut sich großer Bekanntheit"
    ]
    classic_texts = [
        "denn er gilt als zeitloser Klassiker",
        "denn er ist ein Film, der bis heute relevant geblieben ist",
        "denn er wird seit Jahrzehnten geschätzt"
    ]
    modern_texts = [
        "denn er bringt moderne Themen auf die Leinwand",
        "denn er ist ein aktuellerer Film mit frischem Stil",
        "denn er greift zeitgemäße Inhalte auf"
    ]

    if genre_sim > 0.65: reasons.append(random.choice(genre_high))
    elif genre_sim > 0.4: reasons.append(random.choice(genre_mid))
    if tag_sim > 0.4 and tags_selected: reasons.append(random.choice(tag_texts))
    if rating >= 4.0: reasons.append(random.choice(rating_high))
    elif rating >= 3.6: reasons.append(random.choice(rating_mid))
    if n_ratings and n_ratings >= 5000: reasons.append(random.choice(popular_texts))
    if year and year > 2010: reasons.append(random.choice(modern_texts))
    elif year and year < 2000: reasons.append(random.choice(classic_texts))

    trust = float(movie_row.get("similarity", 0))
    trust_percent = round(trust * 100, 1)
    trust_label = "sehr hoch" if trust >= 0.8 else "hoch" if trust >= 0.6 else "mittel"
    vt = f"🔒 Vertrauenswert: {trust_percent}% ({trust_label})"

    if reasons:
        return "Dieser Film wurde empfohlen, " + " und ".join(reasons[:3]) + ". " + vt
    return "Dieser Film passt zu deinem Profil. " + vt

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

    # Auswahlphase (Grid sichtbar, bis 5 Filme gewählt)
    if len(st.session_state.selected_titles) < 5:
        search = st.text_input("🔎 Film suchen oder aus Liste wählen:")
        movies_view = movies[movies["year"] >= min_year].copy()
        available_movies = movies_view.sort_values("title")
        if search:
            available_movies = available_movies[available_movies["title"].str.contains(search, case=False, na=False)]

        page_size = 25
        total_pages = max(1, (len(available_movies) - 1) // page_size + 1)
        start = st.session_state.search_page * page_size
        end = start + page_size
        page_movies = available_movies.iloc[start:end]

        # Zeilenweises Grid: 5 Karten pro Zeile → keine „Lücken“
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

    # Empfehlungsphase (Auswahl ausgeblendet)
    else:
        st.success("✅ Du hast 5 Filme ausgewählt – hier deine Empfehlungen:")

        with st.expander("🔖 Optional: Tags auswählen"):
            all_tags = genome_tags["tag"].astype(str).sort_values().unique().tolist()
            tags_selected = st.multiselect("Bis zu 5 Tags:", all_tags, max_selections=5)

        sel_key = selection_hash(st.session_state.selected_titles, tags_selected, int(min_year))
        if st.session_state.selection_key != sel_key:
            st.session_state.selection_key = sel_key
            st.session_state.rec_index = 3

        # IDs aus vollständigem Datensatz (nicht aus gefiltertem View)
        selected_ids = movies.loc[movies["title"].isin(st.session_state.selected_titles), "movieId"] \
                           .dropna().astype(int).values

        # Genre-Profil (Full) -> Vergleich (View nach Jahr)
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

        # Gesamtscore
        w_tag = 0.5 if tags_selected else 0.0
        w_genre = 1.0 - w_tag
        movies_view_rec["similarity"] = w_genre*movies_view_rec["genre_similarity"] + w_tag*movies_view_rec["tag_similarity"]

        # eigene Auswahl ausschließen
        sorted_movies = movies_view_rec.loc[~movies_view_rec["movieId"].isin(selected_ids)] \
                                       .sort_values("similarity", ascending=False) \
                                       .reset_index(drop=True)

        max_n = len(sorted_movies)
        show_n = min(st.session_state.rec_index, max_n)
        to_show = sorted_movies.iloc[:show_n]

        st.markdown("<h3 class='section-title'>🌟 Empfehlungen</h3>", unsafe_allow_html=True)

        # >>> Empfehlungs-Grid als HTML-Grid mit 3 Spalten <<<
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

        # Mehr laden
        can_more = show_n < max_n
        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            if st.button("🔄 Mehr Empfehlungen laden", disabled=not can_more, use_container_width=True):
                st.session_state.rec_index = min(st.session_state.rec_index + 3, max_n)
                st.rerun()
        if not can_more and max_n > 0:
            st.caption("🎉 Du hast alle passenden Empfehlungen gesehen. Ändere deine Auswahl oder Tags für neue Vorschläge.")


