import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re


def preprocess_text(df):
    df = df.lower()  # Convert to lowercase
    df = re.sub(r'\s+', ' ', df)  # Remove extra spaces
    df= re.sub(r'[^\w\s]', '', df)  # Remove punctuation
    return df

def sentimentScore(df):
    # Plotting the distribution of sentiment scores
    plt.figure(figsize=(12, 6))
    sns.histplot(df['sentiment'], bins=30, kde=True)
    plt.title('Distribution of Sentiment Scores')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show() 

def sentimentClass(df):
    
    #Plot the distribution of the custom sentiment classes.
    plt.figure(figsize=(10, 6))
    sns.countplot(x='sentiment_class', data=df, palette='viridis')
    plt.title('Distribution of Sentiments in Headlines')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Articles')
    plt.show()  