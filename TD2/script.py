import os
import streamlit as st
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag


nltk.download('sentiwordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def load_reviews_from_directory(directory):
    reviews = []
    labels = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                review = file.read()
                reviews.append(review)
                if directory.endswith("pos"):
                    labels.append("positive")
                elif directory.endswith("neg"):
                    labels.append("negative")
    
    return reviews, labels

# Load positive and negative reviews
#positive_reviews, positive_labels = load_reviews_from_directory("C:/Users/aurel/OneDrive - De Vinci/ONE DRIVE PC/A5/NLP/TD2/txt_sentoken/pos")
#negative_reviews, negative_labels = load_reviews_from_directory("C:/Users/aurel/OneDrive - De Vinci/ONE DRIVE PC/A5/NLP/TD2/txt_sentoken/neg")


# Function to calculate sentiment score for a list of words using SentiWordNet
def get_sentiment_score(words):
    total_pos_score = 0
    total_neg_score = 0
    count = 0
    
    for word in words:
        # Lookup sentiment scores in SentiWordNet
        synsets = list(swn.senti_synsets(word))
        if synsets:
            pos_score = synsets[0].pos_score()
            neg_score = synsets[0].neg_score()
            total_pos_score += pos_score
            total_neg_score += neg_score
            count += 1
    
    # Calculate average sentiment scores
    if count > 0:
        avg_pos_score = total_pos_score / count
        avg_neg_score = total_neg_score / count
    else:
        avg_pos_score = 0
        avg_neg_score = 0
    
    return avg_pos_score, avg_neg_score

# Function to classify a review's sentiment
def classify_review_sentiment(review, threshold=0.0):
    sentences = sent_tokenize(review)
    words = []
    
    # Extract words from all sentences
    for sentence in sentences:
        words += word_tokenize(sentence)
    
    # Calculate sentiment scores for words
    avg_pos_score, avg_neg_score = get_sentiment_score(words)
    
    # Determine sentiment based on the threshold
    if avg_pos_score > avg_neg_score + threshold:
        return 'Positive'
    else:
        return 'Negative'

# Function to classify a batch of reviews and calculate accuracy
def classify_reviews_batch(reviews, threshold=0.0, expected_label='Positive'):
    correctly_classified = 0
    
    for review in reviews:
        result = classify_review_sentiment(review, threshold)
        
        if result == expected_label:
            correctly_classified += 1
    
    accuracy = (correctly_classified / len(reviews)) * 100
    return accuracy


# Classify positive reviews and calculate accuracy
#positive_accuracy = classify_reviews_batch(positive_reviews, threshold=0.0, expected_label='positive')

# Classify negative reviews and calculate accuracy
#negative_accuracy = classify_reviews_batch(negative_reviews, threshold=0.0, expected_label='negative')

# Calculate overall accuracy as the average of positive and negative accuracy
#overall_accuracy = (positive_accuracy + negative_accuracy) / 2

#print("Positive Reviews Accuracy: {:.2f}%".format(positive_accuracy))
#print("Negative Reviews Accuracy: {:.2f}%".format(negative_accuracy))
#print("Overall Accuracy: {:.2f}%".format(overall_accuracy))


# Streamlit app
# Specify NLTK data path
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_path)

st.title("WP2 Aurélien Pouxviel : Movie Review Sentiment Analysis App")

review = st.text_area("Enter a movie review :) ")
threshold = st.slider("Sentiment Threshold (0.0 is ok)", min_value=-1.0, max_value=1.0, step=0.01, value=0.0)

if st.button("Classify"):
    if review:
        sentiment = classify_review_sentiment(review, threshold)
        st.write(f"Predicted Sentiment: {sentiment}")
    else:
        st.warning("Please enter a movie review !!! ")