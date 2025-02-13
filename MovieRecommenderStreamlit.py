import streamlit as st
# Set page config must be the first Streamlit command
st.set_page_config(layout="wide", page_title="Movie Recommender")

import pandas as pd
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Custom CSS for IMDB-like styling
st.markdown("""
    <style>
        /* Main background and text colors */
        .stApp {
            background-color: #121212;
            color: #ffffff;
        }
        
        /* IMDB yellow accents */
        .stButton button {
            background-color: #F5C518;
            color: #000000;
            font-weight: bold;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #F5C518 !important;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #1F1F1F;
        }
        
        /* Input fields */
        .stTextInput input {
            background-color: #2D2D2D;
            color: white;
            border: 1px solid #F5C518;
        }
        
        /* Multiselect */
        .stMultiSelect {
            background-color: #2D2D2D;
        }
        
        /* Cards */
        .movie-card {
            background-color: #1F1F1F;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #F5C518;
        }
        
        /* Rating style */
        .rating {
            color: #F5C518;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)


# Load data
@st.cache_data
def load_data():
    movies_df = pd.read_csv('movies_data.csv')
    reviews_df = pd.read_csv('reviews_data.csv')
    return movies_df, reviews_df

movies_df, reviews_df = load_data()

# Code 1: Content-based recommendation functions
def prepare_combined_content(df):
    text_columns = ['Title', 'Genres', 'Director', 'Stars', 'Plot_Summary']
    for col in text_columns:
        df[col] = df[col].fillna('').astype(str)
    df['combined_content'] = (
        df['Title'] + ' ' +
        df['Genres'] + ' ' +
        df['Director'] + ' ' +
        df['Stars'] + ' ' +
        df['Plot_Summary']
    )
    return df

def calculate_movie_similarity(df):
    df = prepare_combined_content(df)
    vectorizer = CountVectorizer(stop_words='english')
    content_matrix = vectorizer.fit_transform(df['combined_content'])
    imdb_weights = df['IMDb_Rating'].values.reshape(-1, 1)
    weighted_matrix = content_matrix.multiply(imdb_weights)
    similarity_matrix = cosine_similarity(weighted_matrix)
    return similarity_matrix, df, vectorizer

def calculate_movie_score(user_input, df):
    similarity_matrix, processed_df, vectorizer = calculate_movie_similarity(df)
    filtered_df = processed_df.copy()
    
    # Apply filters
    if user_input.get('Genres'):
        filtered_df = filtered_df[filtered_df['Genres'].str.contains(user_input['Genres'], case=False, na=False)]
    if user_input.get('Director'):
        filtered_df = filtered_df[filtered_df['Director'].str.contains(user_input['Director'], case=False, na=False)]
    if user_input.get('Stars'):
        filtered_df = filtered_df[filtered_df['Stars'].str.contains(user_input['Stars'], case=False, na=False)]
    if user_input.get('MPAA'):
        filtered_df = filtered_df[filtered_df['MPAA'].str.contains(user_input['MPAA'], case=False, na=False)]
    
    if user_input.get('Title'):
        movie_indices = processed_df[
            processed_df['Title'].str.contains(user_input['Title'], case=False, na=False)
        ].index.tolist()
        
        if movie_indices:
            movie_idx = movie_indices[0]
            similarity_scores = similarity_matrix[movie_idx]
            filtered_df['score'] = 0.0
            for orig_idx, score in zip(processed_df.index, similarity_scores):
                if orig_idx in filtered_df.index and orig_idx != movie_idx:
                    filtered_df.loc[orig_idx, 'score'] = score
            filtered_df = filtered_df[filtered_df.index != movie_idx]
            filtered_df = filtered_df.sort_values(by='score', ascending=False).head(5)
    else:
        input_sentence = ' '.join([
            user_input.get('Genres', ''),
            user_input.get('Director', ''),
            user_input.get('Stars', '')
        ])
        input_vector = vectorizer.transform([input_sentence])
        input_similarity = cosine_similarity(input_vector, vectorizer.transform(filtered_df['combined_content']))[0]
        filtered_df['score'] = input_similarity
        filtered_df = filtered_df.sort_values(by='score', ascending=False).head(5)
    
    return filtered_df

# Code 2: Content-based user history recommendation
@st.cache_resource
def prepare_content_based_model():
    count_vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))
    count_matrix = count_vectorizer.fit_transform(movies_df['Genres'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    movie_indices = pd.Series(movies_df.index, index=movies_df['Title']).drop_duplicates()
    return cosine_sim, movie_indices

cosine_sim, movie_indices = prepare_content_based_model()

# Hàm recommend_movies_content
def recommend_movies_content(user, top_n=5):
    try:
        # Lấy danh sách phim người dùng
        user_likes = reviews_df[(reviews_df['Reviewer'] == user)]['Movie']
        
        if len(user_likes) == 0:
            return []

        # Khởi tạo user profile với shape phù hợp
        user_profile = np.zeros(cosine_sim.shape[1])  # Đảm bảo shape đúng
        count = 0
        
        # Tính trung bình similarity scores cho mỗi phim
        for movie in user_likes:
            idx = movie_indices.get(movie, None)
            if idx is not None:
                user_profile += cosine_sim[idx, :]  # Lấy toàn bộ row của similarity matrix
                count += 1
        
        # Normalize user profile nếu có ít nhất 1 phim
        if count > 0:
            user_profile = user_profile / count
        
        # Tạo danh sách scores cho tất cả phim
        movie_scores = list(enumerate(user_profile))
        
        # Sắp xếp theo score giảm dần
        movie_scores = sorted(movie_scores, key=lambda x: x[1], reverse=True)
        
        # Lọc bỏ những phim user đã xem
        user_watched = set(user_likes)
        recommended_indices = []
        for idx, score in movie_scores:
            movie_title = movies_df.iloc[idx]['Title']
            if movie_title not in user_watched and len(recommended_indices) < top_n:
                recommended_indices.append(idx)
            if len(recommended_indices) >= top_n:
                break
        
        return [movies_df.iloc[idx] for idx in recommended_indices]
    
    except Exception as e:
        st.error(f"Error in content recommendation: {str(e)}")
        return []

# Code 3: Collaborative filtering
@st.cache_resource
def prepare_collaborative_model():
    user_ratings_count = reviews_df.groupby('Reviewer')['Rating'].count()
    valid_users = user_ratings_count[user_ratings_count >= 2].index
    filtered_reviews_df = reviews_df[reviews_df['Reviewer'].isin(valid_users)]
    user_movie_matrix = filtered_reviews_df.pivot_table(index='Reviewer', columns='Movie', values='Rating')
    user_movie_matrix = user_movie_matrix.fillna(0)
    user_similarity = cosine_similarity(user_movie_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)
    return user_movie_matrix, user_similarity_df

user_movie_matrix, user_similarity_df = prepare_collaborative_model()

def recommend_movies_collab(user, top_n=5):
    if user not in user_movie_matrix.index:
        print(f"Người dùng {user} không tồn tại.")
        return []
    
    user_ratings = user_movie_matrix.loc[user]
    rated_movies = user_ratings[user_ratings > 0].index
    print(f"Người dùng đã đánh giá {len(rated_movies)} phim.")
    
    similar_users = user_similarity_df[user].sort_values(ascending=False).drop(user)
    weighted_scores = np.dot(user_similarity_df.loc[user], user_movie_matrix.values)
    recommendation_scores = pd.Series(weighted_scores, index=user_movie_matrix.columns)
    recommendation_scores = recommendation_scores.drop(rated_movies)
    
    if recommendation_scores.empty:
        print("Không còn phim nào để đề xuất.")
        return []

    recommended_movies = recommendation_scores.sort_values(ascending=False).head(top_n)
    print(f"Đề xuất {len(recommended_movies)} phim: {recommended_movies.index.tolist()}")

    return [
        movies_df[movies_df['Title'] == movie].iloc[0]
        for movie in recommended_movies.index if len(movies_df[movies_df['Title'] == movie]) > 0
    ]
    
# Updated UI Helper functions
def load_movie_poster(url):
    try:
        response = requests.get(url, timeout=5)
        img = Image.open(BytesIO(response.content))
        return img
    except:
        return None

def create_movie_card_compact(movie, key_prefix, is_history=False, user_rating=None):
    # Container for the whole card
    with st.container():
        # Show poster
        img = load_movie_poster(movie['Image_link'])
        if img:
            st.image(img, width=150)
        else:
            st.image("https://via.placeholder.com/150x225?text=No+Image", width=150)
        
        # Show title with original formatting
        st.markdown(f"**{movie['Title']}**")
        
        # If it's in watch history, show rating with stars
        if is_history and user_rating is not None:
            stars = "⭐" * int(user_rating)  # Multiply star emoji by rating
            st.markdown(f"**Your Rating:** {stars} ({user_rating:.1f})")
        else:
            # Show details button for non-history items
            if st.button("Show Details", key=f"{key_prefix}_{movie['Title']}"):
                show_movie_details(movie)

def show_movie_details(movie):
    # Create a modal-like effect using columns
    with st.container():
        st.markdown("---")  # Separator
        col1, col2 = st.columns([1, 3])
        
        with col1:
            img = load_movie_poster(movie['Image_link'])
            if img:
                st.image(img, width=200)
            else:
                st.image("https://via.placeholder.com/200x300?text=No+Image", width=200)
        
        with col2:
            st.subheader(movie['Title'])
            st.write(f"**Year:** {movie['Year']}")
            st.write(f"**Genre:** {movie['Genres'].replace(',', ', ')}")
            st.write(f"**Director:** {movie['Director'].replace(',', ', ')}")
            st.write(f"**Stars:** {movie['Stars'].replace(',', ', ')}")
            st.write("**Plot Summary:**")
            st.write(movie['Plot_Summary'])
        st.markdown("---")  # Separator

def display_movie_grid(movies, num_cols=5, key_prefix="grid", is_history=False, ratings_dict=None):
    # Calculate number of rows needed
    n_movies = len(movies)
    n_rows = (n_movies + num_cols - 1) // num_cols
    
    for row in range(n_rows):
        cols = st.columns(num_cols)
        for col in range(num_cols):
            idx = row * num_cols + col
            if idx < n_movies:
                with cols[col]:
                    if isinstance(movies, pd.DataFrame):
                        movie = movies.iloc[idx]
                    else:
                        movie = movies[idx]
                    
                    if is_history:
                        user_rating = ratings_dict.get(movie['Title']) if ratings_dict else None
                        create_movie_card_compact(movie, f"{key_prefix}_{row}_{col}", 
                                               is_history=True, user_rating=user_rating)
                    else:
                        create_movie_card_compact(movie, f"{key_prefix}_{row}_{col}")


# Main execution
if __name__ == "__main__":
    st.title("🎬 Movie Recommender System")
    
    # Sidebar for user login and watch history
    with st.sidebar:
        st.header("User Profile")
        username = st.text_input("Enter your username (ex: redryan64)")
        
        if username:
            st.subheader("Movies You've Watched")
            user_watched = reviews_df[reviews_df['Reviewer'] == username]
            if not user_watched.empty:
                # Get the full movie details for each watched movie
                watched_movies = []
                ratings_dict = dict(zip(user_watched['Movie'], user_watched['Rating']))
                for _, row in user_watched.iterrows():
                    movie_details = movies_df[movies_df['Title'] == row['Movie']].iloc[0]
                    watched_movies.append(movie_details)
                
                # Display watched movies in a grid with ratings
                display_movie_grid(watched_movies, num_cols=2, key_prefix="history", 
                                is_history=True, ratings_dict=ratings_dict)
            else:
                st.write("No watching history found")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        genres = st.multiselect(
            "Genres",
            options=sorted(movies_df['Genres'].str.split(',').explode().unique()),
            key="genres_select"
        )
    
    with col2:
        directors = st.multiselect(
            "Directors",
            options=sorted(movies_df['Director'].dropna().unique()),
            key="directors_select"
        )
    
    with col3:
        stars = st.multiselect(
            "Stars",
            options=sorted(movies_df['Stars'].str.split(',').explode().unique()),
            key="stars_select"
        )
    
    with col4:
        mpaa = st.multiselect(
            "MPAA Rating",
            options=sorted(movies_df['MPAA'].dropna().unique()),
            key="mpaa_select"
        )
    
    # Search box
    search_query = st.text_input("🔍 Search for movies", key="search_input")
    
    # User input dictionary
    user_input = {
        'Title': search_query,
        'Genres': ','.join(genres) if genres else '',
        'Director': ','.join(directors) if directors else '',
        'Stars': ','.join(stars) if stars else '',
        'MPAA': ','.join(mpaa) if mpaa else ''
    }
    
    # Content-based recommendations (Always show)
    st.header("🎯 Recommended Films")
    with st.spinner('Finding recommendations...'):
        recommended_movies = calculate_movie_score(user_input, movies_df)
        if not recommended_movies.empty:
            display_movie_grid(recommended_movies, key_prefix="recommended")
        else:
            st.write("No recommendations found based on your criteria")
    
    # Show personalized recommendations if username is provided
    if username:
        # Content-based recommendations based on user history
        st.header("👍 Movies You May Like")
        with st.spinner('Finding personalized recommendations...'):
            user_recommendations = recommend_movies_content(username)
            if user_recommendations:
                display_movie_grid(user_recommendations, key_prefix="personal")
            else:
                st.write("No personalized recommendations available")
        
        # Collaborative filtering recommendations
        st.header("👥 Movies Everyone Also Likes")
        with st.spinner('Finding collaborative recommendations...'):
            collab_recommendations = recommend_movies_collab(username)
            if collab_recommendations:
                display_movie_grid(collab_recommendations, key_prefix="collab")
            else:
                st.write("No collaborative recommendations available")
            
