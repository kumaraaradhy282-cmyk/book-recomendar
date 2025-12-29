import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Book Recommendation Engine",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Book Recommendation Engine")
st.write("Select a book and get similar book recommendations.")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("books.csv")

books = load_data()

# ---------------- DATA PROCESSING ----------------
books["content"] = books["genre"] + " " + books["description"]

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(books["content"])

similarity_matrix = cosine_similarity(tfidf_matrix)

# ---------------- UI ----------------
book_titles = books["title"].tolist()
selected_book = st.selectbox("Choose a book", book_titles)

# ---------------- RECOMMENDATION FUNCTION ----------------
def recommend_books(book_title, num_recommendations=5):
    index = books[books["title"] == book_title].index[0]
    similarity_scores = list(enumerate(similarity_matrix[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_books = similarity_scores[1:num_recommendations+1]

    recommendations = []
    for i, score in top_books:
        recommendations.append({
            "title": books.iloc[i]["title"],
            "author": books.iloc[i]["author"],
            "genre": books.iloc[i]["genre"]
        })
    return recommendations

if st.button("Recommend"):
    results = recommend_books(selected_book)

    st.subheader("📖 Recommended Books")
    for book in results:
        st.markdown(
            f"**{book['title']}**  \n"
            f"Author: {book['author']}  \n"
            f"Genre: {book['genre']}"
        )
