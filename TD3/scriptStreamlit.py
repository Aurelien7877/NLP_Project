import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os
import sys
import numpy as np
sys.path.insert(0,'C:/Users/aurel/OneDrive - De Vinci/ONE DRIVE PC/A5/NLP/TD3/loadCNN.py')
from loadCNN import loadCNN


articles, abstracts = loadCNN()
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
