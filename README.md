# 🎬 Movie Recommendation System

A smart movie recommendation system that suggests similar movies based on your favorites. Built with FAISS vector search and features 10,000+ Hollywood and Bollywood movies.

## 🚀 [Live Demo](your-streamlit-app-link)

---

## Features

- **Smart Recommendations**: Get personalized movie suggestions based on content similarity and ratings
- **Fuzzy Search**: Find movies even with typos or partial titles
- **Filter by Industry**: Choose between Hollywood, Bollywood, or all movies
- **Movie Posters**: Visual interface with poster images
- **Fast Search**: Instant results powered by FAISS

---

## Tech Stack

- **Python** - Core programming language
- **Streamlit** - Web interface
- **FAISS** - Fast similarity search
- **Hugging Face** - Large file storage
- **OMDB API** - Movie posters

---

## How It Works

1. Enter a movie title you like
2. The system finds similar movies using content analysis (genre, cast, plot, etc.)
3. Results are ranked by both similarity and ratings
4. Get personalized recommendations instantly

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/ArmanKhan-git/Movie-Recommendation-System-FAISS-Index-.git
cd Movie-Recommendation-System-FAISS-Index
```

### 2. Install requirements
```bash
pip install -r requirements.txt
```

### 3. Add your OMDB API key
Create `.streamlit/secrets.toml`:
```toml
OMDB_API_KEY = "your_key_here"
```
Get free key from [OMDB API](http://www.omdbapi.com/apikey.aspx)

### 4. Run the app
```bash
streamlit run app.py
```

---

## Project Links

- **Live App**: [Demo Link](your-streamlit-link)
- **Model Files**: [Hugging Face](https://huggingface.co/ArmanKhan01/Movie-Recommendation-FAISS)

---


## Author

**Your Name**
- GitHub: [@your-username](https://github.com/ArmanKhan-git)

---

## Acknowledgments

- OMDB API for movie data
- Hugging Face for hosting
- Streamlit for the framework

---

⭐ **Star this repo if you like it!**