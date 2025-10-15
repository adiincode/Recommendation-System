# -----------------------------------------------
# 🎥 Movie Recommendation System using NumPy + CSV Dataset
# Author: Aditya Maurya
# -----------------------------------------------

import numpy as np
import pandas as pd

# -------------------------------
# Step 1: Load Dataset
# -------------------------------

# If you have your own dataset, replace this with:
# df = pd.read_csv("your_dataset.csv")

# Sample dataset (movie ratings)
data = {
    "userId": [1, 1, 1, 2, 2, 3, 3, 4, 4, 5],
    "movie": [
        "Inception", "Interstellar", "The Dark Knight",
        "Inception", "Memento",
        "Tenet", "Memento",
        "Interstellar", "Tenet",
        "Inception"
    ],
    "rating": [5, 4, 5, 4, 3, 5, 4, 5, 4, 5]
}

df = pd.DataFrame(data)
print("🎞️ Sample Movie Ratings Dataset:\n")
print(df)

# -------------------------------
# Step 2: Create Pivot Table (User-Movie Rating Matrix)
# -------------------------------

ratings_matrix = df.pivot_table(index="userId", columns="movie", values="rating").fillna(0)
movies = ratings_matrix.columns.tolist()

print("\n🎬 User-Movie Rating Matrix:\n")
print(ratings_matrix)

# Convert pandas DataFrame to numpy array
R = ratings_matrix.to_numpy()

# -------------------------------
# Step 3: Define Cosine Similarity Function
# -------------------------------

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)

# -------------------------------
# Step 4: Compute Movie-to-Movie Similarity Matrix
# -------------------------------

num_movies = len(movies)
similarity_matrix = np.zeros((num_movies, num_movies))

for i in range(num_movies):
    for j in range(num_movies):
        similarity_matrix[i][j] = cosine_similarity(R[:, i], R[:, j])

print("\n🎥 Movie Similarity Matrix (rounded):")
print(np.round(similarity_matrix, 2))

# -------------------------------
# Step 5: Recommendation Function
# -------------------------------

def recommend(movie_name, movies, similarity_matrix, top_n=3):
    if movie_name not in movies:
        print("❌ Movie not found in dataset!")
        return

    movie_idx = movies.index(movie_name)
    similarity_scores = list(enumerate(similarity_matrix[movie_idx]))

    # Sort by similarity score (excluding the same movie)
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    print(f"\n🎯 Because you liked '{movie_name}', you may also like:")
    for idx, score in sorted_scores:
        print(f"  ➤ {movies[idx]} (Similarity: {score:.2f})")

# -------------------------------
# Step 6: Test Recommendation System
# -------------------------------

recommend("Inception", movies, similarity_matrix)
# Try other examples:
# recommend("Tenet", movies, similarity_matrix)
# recommend("Memento", movies, similarity_matrix)
