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

st.title("WP3 :  Aurelien Pouxviel : Article Similarity App")

user_input = st.text_input("Enter some words to find similar articles:")
if user_input:

    user_input_vector = tfidf_vectorizer.transform([user_input])

    user_cosine_similarities = linear_kernel(user_input_vector, article_tfidf_matrix)

    most_similar_article_index = np.argmax(user_cosine_similarities)

    st.subheader(f"Most Similar Article:")
    st.write(articles[most_similar_article_index])
    st.subheader(f"Similarity Score:")
    st.write(np.max(user_cosine_similarities))
