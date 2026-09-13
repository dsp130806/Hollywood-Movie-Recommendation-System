import pandas as pd
import ast
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- Load data ----
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

credits.columns = ['id', 'title', 'cast', 'crew']
movies = movies.merge(credits, on='title')
movies = movies.drop_duplicates(subset='title')

movies = movies[['id_x', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies = movies.rename(columns={'id_x': 'id'})

# ---- Clean columns ----
def extract_names(text):
    try:
        items = ast.literal_eval(text)
        return [item['name'] for item in items]
    except:
        return []

def extract_top_cast(text, limit=3):
    try:
        items = ast.literal_eval(text)
        return [item['name'] for item in items[:limit]]
    except:
        return []

def extract_director(text):
    try:
        items = ast.literal_eval(text)
        for item in items:
            if item['job'] == 'Director':
                return item['name']
        return ''
    except:
        return ''

movies['genres'] = movies['genres'].apply(extract_names)
movies['keywords'] = movies['keywords'].apply(extract_names)
movies['cast'] = movies['cast'].apply(extract_top_cast)
movies['crew'] = movies['crew'].apply(extract_director)

movies['overview'] = movies['overview'].fillna('')
movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew'].apply(lambda x: [x])
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))
movies['tags'] = movies['tags'].apply(lambda x: x.lower())

# ---- Vectorize + similarity ----
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = vectorizer.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

# ---- Recommend function ----
def recommend(movie_title):
    movie_title = movie_title.lower()
    matches = movies[movies['title'].str.lower() == movie_title]

    if matches.empty:
        return "Movie not found in database. Check spelling!"

    idx = matches.index[0]
    distances = similarity[idx]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommendations = []
    for i in movie_list:
        recommendations.append(movies.iloc[i[0]]['title'])

    return recommendations

def recommend_ui(movie_title):
    result = recommend(movie_title)
    if isinstance(result, str):
        return result
    return "\n".join(result)

# ---- Gradio interface ----
interface = gr.Interface(
    fn=recommend_ui,
    inputs=gr.Textbox(placeholder="Enter a movie name, e.g. Avatar"),
    outputs="text",
    title="Hollywood Movie Recommendation System",
    description="Enter a movie you like, and get 5 similar movie suggestions!"
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
