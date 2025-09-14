import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load dataset
st.title("🎬 Movie Recommender System with Multi-Metric Evaluation")
data = pd.read_csv("ratings.csv")  # MovieLens ratings
movies = pd.read_csv("movies.csv")

# Prepare data for Surprise
reader = Reader(rating_scale=(0.5, 5.0))
dataset = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)

trainset, testset = train_test_split(dataset, test_size=0.2)

# Model (Matrix Factorization)
model = SVD()
model.fit(trainset)
predictions = model.test(testset)

# Evaluate
rmse = accuracy.rmse(predictions, verbose=False)
mae = accuracy.mae(predictions, verbose=False)

st.write(f"Model Evaluation:")
st.write(f"- RMSE: {rmse:.3f}")
st.write(f"- MAE: {mae:.3f}")

# Recommend movies for a user
st.subheader("Get Recommendations")
user_id = st.number_input("Enter User ID:", min_value=1, max_value=data['userId'].max(), step=1)

def get_top_n(predictions, n=5):
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        top_n.setdefault(uid, [])
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

top_n = get_top_n(predictions, n=5)

if user_id in top_n:
    st.write("Top 5 Recommended Movies:")
    for movie_id, rating in top_n[user_id]:
        title = movies[movies['movieId'] == movie_id]['title'].values[0]
        st.write(f"- {title} (Predicted Rating: {rating:.2f})")
else:
    st.write("No recommendations available for this user (cold start).")
