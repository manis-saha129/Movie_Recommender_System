# streamlit_movie_recommender.py
import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import nltk

nltk.download('punkt')


# Load data with caching
@st.cache_data
def load_data():
    movies = pd.read_csv('tmdb_5000_movies.csv/tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv/tmdb_5000_credits.csv')
    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies.dropna(inplace=True)
    return movies


movies = load_data()


# Helper functions for data preprocessing
def convert(obj):
    try:
        return [i['name'] for i in ast.literal_eval(obj)]
    except Exception as e:
        return []


def convert3(obj):
    try:
        return [i['name'] for i in ast.literal_eval(obj)[:3]]
    except Exception as e:
        return []


def fetch_director(obj):
    try:
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                return [i['name']]
        return []
    except Exception as e:
        return []


# Apply conversion functions with caching
@st.cache_data
def preprocess_movies(movies):
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    new_df = movies[['movie_id', 'title', 'tags']]
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
    return new_df


new_df = preprocess_movies(movies)


# Text stemming with caching
@st.cache_data
def preprocess_tags(new_df):
    ps = PorterStemmer()

    def stem(text):
        return " ".join([ps.stem(i) for i in text.split()])
    new_df['tags'] = new_df['tags'].apply(stem).apply(lambda x: x.lower())
    return new_df


new_df = preprocess_tags(new_df)


# Create count matrix and similarity with caching
@st.cache_data
def compute_similarity(new_df):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    similarity = cosine_similarity(vectors)
    return similarity


similarity = compute_similarity(new_df)


# Recommendation function
def recommend(movie):
    try:
        movie_index = new_df[new_df['title'] == movie].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        recommended_movies = [new_df.iloc[i[0]].title for i in movies_list]
        return recommended_movies
    except IndexError:
        return []


# Streamlit app
st.title('Movie Recommendation System')

selected_movie_name = st.selectbox(
    'Select a movie you like:',
    movies['title'].values
)

if st.button('Recommend'):
    recommendations = recommend(selected_movie_name)
    st.write('**Recommended movies:**')
    for movie in recommendations:
        st.write(movie)
