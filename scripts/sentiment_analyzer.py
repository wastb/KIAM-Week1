from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rake_nltk import Rake
nltk.download('punkt_tab')
nltk.download('stopwords')
import re

def preprocess_text(text):
    """
    Preprocess text by converting to lowercase, removing extra spaces, and removing punctuation.
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

def sentiment_category(df):
    sia = SentimentIntensityAnalyzer()
    headlines = df['headline'].dropna().tolist()
    preprocessed_headlines = [preprocess_text(headline) for headline in headlines]

    sentiment_results = []
    sentiment_score = []
    for i in preprocessed_headlines:
        polarity_score = sia.polarity_scores(i)
        if polarity_score['compound'] >= 0.05:
          sentiment_results.append('Positive')
          sentiment_score.append(polarity_score['compound'])
        if polarity_score['compound']<= -0.05:
          sentiment_results.append('Negative')
          sentiment_score.append(polarity_score['compound'])
        if -0.05 < polarity_score['compound'] < 0.05:
          sentiment_results.append('Neutral')
          sentiment_score.append(polarity_score['compound'])

    df['sentiment_class'] = sentiment_results
    df['sentiment_score'] = sentiment_score
    return df

def ranked_phrases(df):
    r = Rake(include_repeated_phrases=False)
    headlines = df['headline'].dropna().tolist()
    preprocessed_headlines = [preprocess_text(headline) for headline in headlines]
    ranked_phrases = []

    for i in preprocessed_headlines:
       r.extract_keywords_from_text(i)
       
       for score, phrase in r.get_ranked_phrases_with_scores():
            ranked_phrases.append((score, phrase))
    
    key_phrases = pd.DataFrame(ranked_phrases, columns = ['score','phrase'])
    return key_phrases

def sentiment_score_dist(df):
    # Plotting the distribution of sentiment scores
    plt.figure(figsize=(12, 6))
    sns.histplot(df['sentiment_score'], bins=30, kde=True)
    plt.title('Distribution of Sentiment Scores')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show();

def sentiment_class_plot(df):
    
    #Plot the distribution of the custom sentiment classes.
    plt.figure(figsize=(10, 6))
    sns.countplot(x='sentiment_class', data=df)
    plt.title('Distribution of Sentiments in Headlines')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Articles')
    plt.show();  

def plot_scatter(final_data):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=final_data, x='average_sentiment_score', y='daily_return')
    plt.title('Sentiment Score vs. Daily Return')
    plt.xlabel('Average Sentiment Score')
    plt.ylabel('Daily Return')

    plt.subplot(1, 2, 2)
    sns.scatterplot(data=final_data, x='average_sentiment_score', y='Close')
    plt.title('Sentiment Score vs. Closing Price')
    plt.xlabel('Average Sentiment Score')
    plt.ylabel('Closing Price')

    plt.tight_layout()
    plt.show()

def plot_heatmap(final_data):
    correlation_matrix = final_data[['average_sentiment_score', 'daily_return', 'Close']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.show()

def plot_distributions(final_data):
    plt.figure(figsize=(14, 7))
    plt.subplot(2, 2, 1)
    sns.histplot(final_data['Close'], kde=True, bins=30)
    plt.title('Distribution of Closing Prices')
    plt.xlabel('Closing Price')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 2, 2)
    sns.histplot(final_data['average_sentiment_score'], kde=True, bins=30)
    plt.title('Distribution of Sentiment Scores')
    plt.xlabel('Average Sentiment Score')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

def plot_scatter_and_heatmap(final_data):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=final_data, x='average_sentiment_score', y='daily_return')
    plt.title('Sentiment Score vs. Daily Return')
    plt.xlabel('Average Sentiment Score')
    plt.ylabel('Daily Return')

    plt.subplot(1, 2, 2)
    sns.scatterplot(data=final_data, x='average_sentiment_score', y='Close')
    plt.title('Sentiment Score vs. Closing Price')
    plt.xlabel('Average Sentiment Score')
    plt.ylabel('Closing Price')

    plt.tight_layout()
    plt.show()

    correlation_matrix = final_data[['average_sentiment_score', 'daily_return', 'Close']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.show()


def key_phrases_plot(df):
    phrases = ranked_phrases(df).nlargest(20,'score')
    plt.figure(figsize=(14, 8))
    sns.barplot(x= phrases['score'], y=phrases['phrase'], data=phrases)
    plt.title('Top 20 Keywords by TF-IDF Score')
    plt.xlabel('TF-IDF Score')
    plt.ylabel('Keyword')
    plt.show();