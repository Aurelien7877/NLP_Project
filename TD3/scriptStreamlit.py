import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os
import sys
import numpy as np


import pickle


# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Create file paths relative to the script's location
articles_path = os.path.join(script_dir, 'CNNArticles')
abstracts_path = os.path.join(script_dir, 'CNNGold')


# Open and load the data
with open(articles_path, 'rb') as file:
    articles = pickle.load(file)

with open(abstracts_path, 'rb') as file:
    abstracts = pickle.load(file)

articlesCl = []  
for article in articles:
    articlesCl.append(article.replace("”", "").rstrip("\n"))
articles = articlesCl
    
articlesCl = []  
for article in abstracts:
    articlesCl.append(article.replace("”", "").rstrip("\n"))
abstracts = articlesCl
    

tfidf_vectorizer = TfidfVectorizer()
article_tfidf_matrix = tfidf_vectorizer.fit_transform(articles)
summary_tfidf_matrix = tfidf_vectorizer.transform(abstracts)

# Title with colored background and special font
st.markdown(
    "<h1 style='text-align: center; background-color: #F63366; color: white;"
    "font-size: 32px; padding: 10px; border-radius: 10px;'>WP3: Aurelien Pouxviel: Article Similarity App</h1>",
    unsafe_allow_html=True,
)

user_input = st.text_input("💡 Enter some words to find similar articles ​💡:")

# Button to trigger the action
if st.button("Find Similar Articles 🔎​🤔​"):
    if user_input:
        # Loading bar
        with st.spinner("Finding similar articles..."):
            user_input_vector = tfidf_vectorizer.transform([user_input])
            user_cosine_similarities = linear_kernel(user_input_vector, article_tfidf_matrix)

            # Find the top 2 similar articles
            most_similar_article_indices = np.argsort(user_cosine_similarities[0])[-2:][::-1]

        st.subheader("💎​Top 2​ Most Similar Articles 💎​ ")

        for i, index in enumerate(most_similar_article_indices):
            st.subheader(f"Article {i + 1} (Similarity Score: {user_cosine_similarities[0][index]:.2f}):")
            st.write(articles[index])

        st.success("Search completed! ✅")
