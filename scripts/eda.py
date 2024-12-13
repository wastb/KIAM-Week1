import os
import talib
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
import matplotlib.pyplot as plt
import pynance

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def summary_statistics(df):

    return df.describe()


def check_missing_values(df):

    return df.isnull().sum()

def box_plot(df):
    