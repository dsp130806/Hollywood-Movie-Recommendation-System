# 🎬 Hollywood Movie Recommendation System

A content-based movie recommendation system that suggests similar movies based on genre, plot, cast, and director — built using Python, scikit-learn, and TF-IDF vectorization.

## 📌 Overview

This project recommends movies similar to a given title by analyzing:
- Plot overview
- Genres
- Keywords
- Top 3 cast members
- Director

All this information is combined into a single "tag" per movie, converted into numerical vectors using **TF-IDF**, and compared using **Cosine Similarity** to find the most similar movies.

## 🛠️ Tech Stack

- Python
- Pandas (data handling)
- Scikit-learn (TF-IDF Vectorizer, Cosine Similarity)
- Gradio (interactive web interface)

## 📊 Dataset

[TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) — contains metadata for 4,803 movies including genres, cast, crew, and plot summaries.

## ⚙️ How It Works

1. Merge movie metadata with cast/crew data
2. Extract genres, keywords, top cast, and director from raw JSON-like fields
3. Combine all features into a single text "tag" per movie
4. Convert tags into numerical vectors using TF-IDF
5. Calculate cosine similarity between all movie pairs
6. Given a movie title, return the top 5 most similar movies

## 🎯 Example Results

**Input:** Avatar
**Output:** Aliens, Alien, Moonraker, Alien³, Silent Running

**Input:** The Dark Knight Rises
**Output:** The Dark Knight, Batman Begins, Batman Returns, Batman, Batman Forever

## ⚠️ Limitations

- Dataset is primarily Hollywood/English-language films — recommendations for non-English or regional cinema (e.g., Tollywood) are unreliable or unavailable, since those films are barely represented in this dataset.
- Recommendations are based on content similarity (genre, cast, plot themes), not user ratings or viewing behavior — this is a **content-based** recommender, not a collaborative filtering one.

## 🚀 Try It Yourself

Open the notebook in Google Colab (link included in the notebook file) and run all cells. A Gradio interface will launch where you can type any movie title and get instant recommendations.

## 📈 Future Improvements

- Build a separate recommender for Tollywood/regional cinema using a custom dataset
- Add collaborative filtering using user rating data
- Deploy as a standalone web app (beyond just the Colab-based Gradio demo)
