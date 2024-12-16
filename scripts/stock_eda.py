import os
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
import matplotlib.pyplot as plt

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def summary_statistics(df):

    return df.describe()


def check_missing_values(df):

    return df.isnull().sum()

def convert_date(df):

    """Convert the 'Date' column to datetime and set it as the index."""
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def box_plot(df):
# Plot multiple box plots
  plt.figure(figsize=(10, 6))  # Adjust the figure size
  sns.boxplot(data=df, orient="h")  # Horizontal box plots for better readability

# Add labels and title
  plt.title("Box Plots for Outlier Detection", fontsize=16)
  plt.xlabel("Value", fontsize=12)
  plt.ylabel("Columns", fontsize=12)

# Show the plot
  plt.show()
