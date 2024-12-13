#import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def missing_values(df):

    return df.isnull().sum()

def remove_columns(df):

    return df.drop(columns=['Unnamed: 0'])

def summary_statistics(df):

    return df.describe()

def convert_date(df):

    """Convert the 'Date' column to datetime and set it as the index."""

    date_length = df['date'].apply(len)
    
    # Truncate the 'date' column to a length of 19 characters
    df['date'] = df['date'].str.slice(0, 19)   
    # Convert the 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    return df

def headline_length(df):

    df['headline_length'] = df['headline'].apply(len)
    return df

def topPublisher(df):

    top_publishers = df['publisher'].value_counts().head(20)
    return top_publishers

def top_publisher_plot(df):
    top_publishers = df['publisher'].value_counts().head(20)
    plt.figure(figsize=(14, 8))
    sns.barplot(x=top_publishers.index, y=top_publishers.values, palette="viridis")
    plt.xticks(rotation=90)
    plt.title('Top 20 Publishers by Number of Articles')
    plt.xlabel('Publisher')
    plt.ylabel('Number of Articles')
    plt.show();

def extract_dates(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['day_of_week'] = df['date'].dt.day_name()

    return df
#Number of Articles Published per Year

def yearly_published_article(df):
    plt.figure(figsize=(12, 6))
    articles_per_year = df.groupby('year').size()
    plt.plot(articles_per_year.index, articles_per_year.values, marker='o')
    plt.title('Number of Articles Published per Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Articles')
    plt.grid(True)
    plt.show();


def publication_per_day(df):
    # Filter data for January of 2017, 2018, and 2019
    years = [2017, 2018, 2019,2020]
    df_filtered = df[df['year'].isin(years) & (df['month'] == 1)]

    # Count the number of articles per day for each year
    daily_counts = df_filtered.groupby(['year', 'day']).size().reset_index(name='count')

    # Plotting the number of articles published per day for January in each year
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=daily_counts, x='day', y='count', hue='year', marker='o')
    plt.title('Number of Articles Published Per Day in January (2018-2020)')
    plt.xlabel('Day')
    plt.ylabel('Number of Articles')
    plt.legend(title='Year')
    plt.grid(True)
    plt.xticks(daily_counts['day'].unique())  # Ensure all days are shown on x-axis
    plt.show();

def publication_time(df):
    # Filter data for January of 2017, 2018, and 2019
    

    # Count the number of articles per day for each year
    daily_counts = df.groupby(['hour']).size().reset_index(name='count')

    # Plotting the number of articles published per day for January in each year
    plt.figure(figsize=(14, 8))
    sns.barplot(data=daily_counts, x='hour', y='count')
    plt.title('Number of Articles Published Every Hour')
    plt.xlabel('Hour')
    plt.ylabel('Number of Articles')
    plt.grid(True)
    plt.xticks(daily_counts['hour'].unique())  # Ensure all days are shown on x-axis
    plt.show();

def publication_day(df):
    # Filter data for January of 2017, 2018, and 2019
    

    # Count the number of articles per day for each year
    daily_counts = df.groupby(['day_of_week']).size().reset_index(name='count')

    # Plotting the number of articles published per day for January in each year
    plt.figure(figsize=(14, 8))
    sns.barplot(data=daily_counts, x='day_of_week', y='count')
    plt.title('Number of Articles Published Every Hour')
    plt.xlabel('day_of_week')
    plt.ylabel('Number of Articles')
    plt.grid(True)
    plt.xticks(daily_counts['day_of_week'].unique())  # Ensure all days are shown on x-axis
    plt.show();

def publication_per_month(df):
    #Extract year and month from the date
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    # Count the number of articles per year and month
    articles_per_month = df.groupby(['year', 'month']).size().reset_index(name='count')

    # Plotting the number of articles per month for each year
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=articles_per_month, x='month', y='count', hue='year', marker='o', palette='tab10')
    plt.title('Number of Articles Published Each Month by Year')
    plt.xlabel('Month')
    plt.ylabel('Number of Articles')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend(title='Year')
    plt.grid(True)
    plt.show();

def publication_over_time(df):
    # Count the number of articles per date
    articles_per_date = df['date'].value_counts().sort_index()

    # Convert to DataFrame for plotting
    articles_per_date_df = articles_per_date.reset_index()
    articles_per_date_df.columns = ['date', 'count']

    # Plotting the number of articles over time
    plt.figure(figsize=(16, 8))
    sns.lineplot(data=articles_per_date_df, x='date', y='count')
    plt.title('Number of Articles Published Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.grid(True)
    plt.show();

def extract_domain(df):
    df['domains'] = df['publisher'].str.extract(r'@([^.]+)')
    return df

def domain_plot(df):
  
  domain_count = df['domains'].value_counts()
  plt.figure(figsize=(12, 6))
  sns.barplot(x=domain_count.index, y=domain_count)
  plt.title('Top Domains by Number of Articles')
  plt.xlabel('Domain')
  plt.ylabel('Number of Articles')
  plt.xticks(rotation=45)
  plt.grid(True)
  plt.tight_layout()
  plt.show();




