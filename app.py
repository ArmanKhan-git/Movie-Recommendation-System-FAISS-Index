import streamlit as st
import joblib
import faiss
import requests
import pandas as pd
import numpy as np
from thefuzz import process
from huggingface_hub import hf_hub_download  # NEW: Add this import

st.set_page_config(layout="wide", page_title="Movie Recommendation System")

# --- CUSTOM CSS FOR UNIFORM POSTER SIZES ---
st.markdown("""
<style>
    .poster-img {
        height: 300px; /* Enforce a fixed height for all posters */
        width: 100%;    /* Make the image fill the column width */
        object-fit: cover; /* Cover the area, cropping if necessary, without distortion */
        border-radius: 10px; /* Optional: Add rounded corners */
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2); /* Optional: Add a subtle shadow */
    }
    .poster-title {
        text-align: center;
        font-size: 14px;
        font-weight: bold;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)


# --- API AND DATA LOADING ---
OMDB_API_KEY = st.secrets["OMDB_API_KEY"]

@st.cache_resource
def load_data():
    """Loads all necessary data and models - downloads large files from HF if needed."""
    
    # Load small files from local (GitHub repo)
    movie_list = joblib.load('moives.pkl')
    industry_indices = joblib.load('industry_indices.pkl')
    
    # UPDATED: Download large files from Hugging Face
    faiss_index_path = hf_hub_download(
        repo_id="ArmanKhan01/Movie-Recommendation-FAISS",
        filename="faiss_index.index",
        repo_type="model"
    )
    
    vectors_path = hf_hub_download(
        repo_id="ArmanKhan01/Movie-Recommendation-FAISS",
        filename="vectors.pkl",
        repo_type="model"
    )
    
    # Load the downloaded files
    index = faiss.read_index(faiss_index_path)
    vectors = joblib.load(vectors_path)
    
    return movie_list, index, vectors, industry_indices

df, index, vectors_dense, industry_indices = load_data()

@st.cache_data
def fetch_poster(movie_title):
    """Fetches a movie poster URL from the OMDB API, with caching."""
    try:
        url = f"http://www.omdbapi.com/?t={movie_title}&apikey={OMDB_API_KEY}"
        response = requests.get(url)
        data = response.json()
        if data.get('Response') == 'True' and 'Poster' in data and data['Poster'] != 'N/A':
            return data['Poster']
        else:
            return "https://via.placeholder.com/200x300.png?text=Poster+Not+Found"
    except Exception as e:
        st.error(f"Error fetching poster: {e}")
        return "https://via.placeholder.com/200x300.png?text=Error"

# --- CORE LOGIC FUNCTIONS ---
def get_closest_matches(query, choices, limit=5):
    results = process.extract(query, choices, limit=limit)
    return [result[0] for result in results if result[1] > 50]

def recommend_faiss_hybrid_filtered(movie_title, k=50, industry='All'):
    alpha = 0.7
    try:
        movie_idx = df[df['title'] == movie_title].index[0]
    except IndexError: return []
    search_params = faiss.SearchParameters()
    if industry != 'All':
        ids_to_search = industry_indices.get(industry)
        if ids_to_search is None or len(ids_to_search) == 0: return []
        selector = faiss.IDSelectorArray(ids_to_search)
        search_params.sel = selector
    query_vector = vectors_dense[movie_idx:movie_idx+1]
    D, I = index.search(query_vector, k, params=search_params)
    candidates = []
    for i, sim_score in zip(I[0], D[0]):
        if i == -1 or i == movie_idx: continue
        wr_norm = df.iloc[i]['wr_norm_score']
        hybrid_score = (alpha * sim_score) + ((1 - alpha) * wr_norm)
        candidates.append({'title': df.iloc[i].title, 'hybrid_score': hybrid_score})
    candidates.sort(key=lambda x: x['hybrid_score'], reverse=True)
    return candidates

# --- STREAMLIT UI & STATE MANAGEMENT ---
st.title('🎬 Movie Recommendation System')
st.markdown("Discover your next favorite movie. Enter a film title, and we'll handle the rest.")

if 'selected_movie' not in st.session_state: st.session_state.selected_movie = None
if 'suggestions' not in st.session_state: st.session_state.suggestions = []
if 'recommendations' not in st.session_state: st.session_state.recommendations = []
if 'num_to_show' not in st.session_state: st.session_state.num_to_show = 5

movie_title_input = st.text_input("Enter a movie title to get started:", placeholder="e.g., The Dark Knight")

if st.button("Get Recommendations", type="primary"):
    st.session_state.selected_movie = None
    st.session_state.recommendations = []
    st.session_state.suggestions = []
    if movie_title_input.strip():
        exact_match = df[df['title'].str.lower() == movie_title_input.strip().lower()]
        if not exact_match.empty:
            st.session_state.selected_movie = exact_match['title'].iloc[0]
        else:
            st.session_state.suggestions = get_closest_matches(movie_title_input, df['title'])
            if not st.session_state.suggestions:
                st.error("No close matches found. Please try a different title.")

if st.session_state.suggestions:
    st.markdown("---")
    st.subheader("Did you mean...?")
    cols = st.columns(len(st.session_state.suggestions))
    for i, movie in enumerate(st.session_state.suggestions):
        with cols[i]:
            if st.button(movie, use_container_width=True):
                st.session_state.selected_movie = movie
                st.session_state.suggestions = []
                st.rerun()

if st.session_state.selected_movie:
    st.markdown("---")
    st.subheader(f" Get Recommendations for '{st.session_state.selected_movie}'")
    col1, col2, col3 = st.columns(3)
    def handle_recommendation_click(industry):
        st.session_state.num_to_show = 5
        with st.spinner(f'Finding {industry} recommendations...'):
            st.session_state.recommendations = recommend_faiss_hybrid_filtered(
                st.session_state.selected_movie, industry=industry)
    with col1:
        if st.button("All Movies", use_container_width=True): handle_recommendation_click('All')
    with col2:
        if st.button("Hollywood Only", use_container_width=True): handle_recommendation_click('Hollywood')
    with col3:
        if st.button("Bollywood Only", use_container_width=True): handle_recommendation_click('Bollywood')

if st.session_state.recommendations:
    recommendations_to_show = st.session_state.recommendations[:st.session_state.num_to_show]
    st.markdown("---")
    st.subheader(f"Here are your top recommendations:")
    cols = st.columns(5)
    for i, rec in enumerate(recommendations_to_show):
        with cols[i % 5]:
            poster_url = fetch_poster(rec['title'])
            st.markdown(f"<img class='poster-img' src='{poster_url}'>", unsafe_allow_html=True)
            st.markdown(f"<p class='poster-title'>{rec['title']}</p>", unsafe_allow_html=True)

    if len(st.session_state.recommendations) > st.session_state.num_to_show:
        st.markdown("<br>", unsafe_allow_html=True)
        col1_more, col2_more, col3_more = st.columns([2, 1, 2])
        with col2_more:
            if st.button("🎬 +5 More Movies", use_container_width=True):
                st.session_state.num_to_show += 5
                st.rerun()