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
st.write("Select a book to get similar book recommendations.")

# ---------------- LOAD DATA ----------------
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("books.csv")
    df = df.fillna("")  # prevent warnings/errors
    return df

books = load_data()

# ---------------- PREPROCESSING ----------------
books = books.copy()  # avoid pandas warnings
books["content"] = (
    books["genre"].astype(str) + " " +
    books["description"].astype(str)
)

# ---------------- VECTORIZATION ----------------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(books["content"])

similarity_matrix = cosine_similarity(tfidf_matrix)

# ---------------- UI ----------------
book_titles = books["title"].tolist()
selected_book = st.selectbox("Choose a book", book_titles)

# ---------------- RECOMMENDATION FUNCTION ----------------
def recommend_books(book_title, top_n=5):
    book_index = books.index[books["title"] == book_title][0]
    similarity_scores = list(enumerate(similarity_matrix[book_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommended_indices = [i for i, _ in similarity_scores[1:top_n + 1]]

    return books.loc[recommended_indices, ["title", "author", "genre"]]

# ---------------- OUTPUT ----------------
if st.button("Recommend Books"):
    recommendations = recommend_books(selected_book)

    st.subheader("📖 Recommended Books")
    for _, row in recommendations.iterrows():
        st.markdown(
            f"**{row['title']}**  \n"
            f"Author: {row['author']}  \n"
            f"Genre: {row['genre']}"
        )
