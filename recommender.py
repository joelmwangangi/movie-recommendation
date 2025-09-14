import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Load Data
# ----------------------------
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

st.title("🎬 Movie Recommender System with Cold Start Simulation")

# ----------------------------
# Create User-Item Matrix
# ----------------------------
user_item_matrix = ratings.pivot_table(
    index="userId", columns="movieId", values="rating"
).fillna(0)

# Compute cosine similarity between movies (item-based)
similarity = cosine_similarity(user_item_matrix.T)
similarity_df = pd.DataFrame(
    similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns
)

# ----------------------------
# Recommendation Function
# ----------------------------
def recommend_movie(movie_id, n=5):
    if movie_id not in similarity_df.columns:
        return pd.DataFrame(columns=["title", "genres"])
    
    similar_scores = similarity_df[movie_id].sort_values(ascending=False)
    top_movies = similar_scores.iloc[1:n+1].index  # skip itself
    return movies[movies["movieId"].isin(top_movies)][["title", "genres"]]

# ----------------------------
# Cold Start Simulation
# ----------------------------
def cold_start_simulation(new_user_ratings, n=5):
    """
    Simulates a cold start user with only a few ratings.
    new_user_ratings: dict {movieId: rating}
    """
    # Add the new user to the user-item matrix
    new_user_df = pd.DataFrame(new_user_ratings, index=[9999])  # dummy user ID
    temp_matrix = pd.concat([user_item_matrix, new_user_df]).fillna(0)

    # Compute cosine similarity for the new user
    user_sim = cosine_similarity(temp_matrix)
    user_sim_df = pd.DataFrame(
        user_sim, index=temp_matrix.index, columns=temp_matrix.index
    )

    # Find most similar existing user
    sim_scores = user_sim_df.loc[9999].drop(9999)
    most_similar_user = sim_scores.idxmax()

    # Recommend based on that user’s top movies
    top_movies = ratings[ratings["userId"] == most_similar_user].sort_values(
        "rating", ascending=False
    ).head(n)["movieId"]
    
    return movies[movies["movieId"].isin(top_movies)][["title", "genres"]]

# ----------------------------
# Streamlit UI
# ----------------------------
st.subheader("🔹 Standard Recommendation (Item-based)")
selected_movie = st.selectbox("Pick a movie you like:", movies["title"])
movie_id = movies[movies["title"] == selected_movie]["movieId"].values[0]

st.write("Because you liked:", selected_movie)
st.write("We recommend:")
st.table(recommend_movie(movie_id))

st.subheader("🔹 Cold Start Simulation (New User)")
st.write("Simulating a new user with very few ratings...")

# Example: new user rates Toy Story = 5, Jumanji = 2
new_user_ratings = {1: 5, 2: 2}
st.json(new_user_ratings)

st.write("Recommendations for this new user:")
st.table(cold_start_simulation(new_user_ratings))

st.subheader("📊 Future Extensions")
st.markdown("""
- Add Precision@K, Recall@K, RMSE for accuracy  
- Compute diversity, novelty, serendipity, fairness, coverage  
- Visualize results with charts
""")
