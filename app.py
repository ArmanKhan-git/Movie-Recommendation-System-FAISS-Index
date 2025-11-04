import streamlit as st
import joblib
import faiss
import requests
import pandas as pd
import numpy as np
from thefuzz import process
from huggingface_hub import hf_hub_download

st.set_page_config(layout="wide", page_title="Movie Recommendation System")

# --- CUSTOM CSS FOR UNIFORM POSTER SIZES ---
st.markdown("""
<style>
    .poster-img {
        height: 300px;
        width: 100%;
        object-fit: cover;
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
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
    
    # Download large files from Hugging Face
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
def get_filtered_movies(query, movie_list, limit=100):
    """Returns movies that contain the search query (case-insensitive)"""
    if not query or len(query.strip()) == 0:
        return []
    
    query_lower = query.lower().strip()
    filtered = [movie for movie in movie_list if query_lower in movie.lower()]
    return sorted(filtered)[:limit]

def recommend_faiss_hybrid_filtered(movie_title, k=50, industry='All'):
    alpha = 0.7
    try:
        movie_idx = df[df['title'] == movie_title].index[0]
    except IndexError: 
        return []
    
    search_params = faiss.SearchParameters()
    if industry != 'All':
        ids_to_search = industry_indices.get(industry)
        if ids_to_search is None or len(ids_to_search) == 0: 
            return []
        selector = faiss.IDSelectorArray(ids_to_search)
        search_params.sel = selector
    
    query_vector = vectors_dense[movie_idx:movie_idx+1]
    D, I = index.search(query_vector, k, params=search_params)
    candidates = []
    
    for i, sim_score in zip(I[0], D[0]):
        if i == -1 or i == movie_idx: 
            continue
        wr_norm = df.iloc[i]['wr_norm_score']
        hybrid_score = (alpha * sim_score) + ((1 - alpha) * wr_norm)
        candidates.append({'title': df.iloc[i].title, 'hybrid_score': hybrid_score})
    
    candidates.sort(key=lambda x: x['hybrid_score'], reverse=True)
    return candidates

# --- STREAMLIT UI & STATE MANAGEMENT ---
st.title('🎬 Movie Recommendation System')
st.markdown("Discover your next favorite movie. Search and select a film title below.")

# Initialize session state
if 'selected_movie' not in st.session_state: 
    st.session_state.selected_movie = None
if 'recommendations' not in st.session_state: 
    st.session_state.recommendations = []
if 'num_to_show' not in st.session_state: 
    st.session_state.num_to_show = 5
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""

# Search input
movie_title_input = st.text_input(
    "🔍 Search for a movie:", 
    placeholder="Start typing... (e.g., Dark Knight, Avengers, 3 Idiots)",
    key="search_input"
)

# Get all movie titles as a list
all_movies = df['title'].tolist()

# Filter movies based on search query
filtered_movies = get_filtered_movies(movie_title_input, all_movies)

# Show selectbox only if there's input and matches found
if movie_title_input.strip() and filtered_movies:
    selected_movie = st.selectbox(
        "📽️ Select a movie from the list:",
        options=[""] + filtered_movies,  # Empty option at start
        format_func=lambda x: "Choose a movie..." if x == "" else x,
        key="movie_selector"
    )
    
    # Show Get Recommendations button when a movie is selected
    if selected_movie and selected_movie != "":
        if st.button("Get Recommendations", type="primary", use_container_width=True):
            st.session_state.selected_movie = selected_movie
            st.session_state.recommendations = []
            st.session_state.num_to_show = 5
            st.rerun()

elif movie_title_input.strip() and not filtered_movies:
    st.warning("No movies found matching your search. Try a different title.")

# Show recommendation options if a movie is selected
if st.session_state.selected_movie:
    st.markdown("---")
    st.subheader(f"📌 Get Recommendations for '{st.session_state.selected_movie}'")
    
    col1, col2, col3 = st.columns(3)
    
    def handle_recommendation_click(industry):
        st.session_state.num_to_show = 5
        with st.spinner(f'Finding {industry} recommendations...'):
            st.session_state.recommendations = recommend_faiss_hybrid_filtered(
                st.session_state.selected_movie, industry=industry)
    
    with col1:
        if st.button("🌍 All Movies", use_container_width=True): 
            handle_recommendation_click('All')
    with col2:
        if st.button("🎬 Hollywood Only", use_container_width=True): 
            handle_recommendation_click('Hollywood')
    with col3:
        if st.button("🎭 Bollywood Only", use_container_width=True): 
            handle_recommendation_click('Bollywood')

# Display recommendations
if st.session_state.recommendations:
    recommendations_to_show = st.session_state.recommendations[:st.session_state.num_to_show]
    st.markdown("---")
    st.subheader(f"✨ Here are your top recommendations:")
    
    cols = st.columns(5)
    for i, rec in enumerate(recommendations_to_show):
        with cols[i % 5]:
            poster_url = fetch_poster(rec['title'])
            st.markdown(f"<img class='poster-img' src='{poster_url}'>", unsafe_allow_html=True)
            st.markdown(f"<p class='poster-title'>{rec['title']}</p>", unsafe_allow_html=True)

    # Show "Load More" button if there are more recommendations
    if len(st.session_state.recommendations) > st.session_state.num_to_show:
        st.markdown("<br>", unsafe_allow_html=True)
        col1_more, col2_more, col3_more = st.columns([2, 1, 2])
        with col2_more:
            if st.button("🎬 +5 More Movies", use_container_width=True):
                st.session_state.num_to_show += 5
                st.rerun()