#import library
import talib
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


def calculate_technical_indicators(df):
    """Calculate basic technical indicators using TA-Lib."""
    df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
    df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    return df

def plot_stock_data(df):
    """Plot the stock data with technical indicators."""
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Close'], label='Close', color='blue')
    plt.plot(df.index, df['SMA_20'], label='20-day SMA', color='red')
    plt.plot(df.index, df['SMA_50'], label='50-day SMA', color='green')
    plt.title('Stock Price and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show();

def plot_rsi(df):

    """Plot the Relative Strength Index (RSI)."""

    plt.figure(figsize=(14, 5))
    plt.plot(df.index, df['RSI'], label='RSI', color='purple')
    plt.axhline(70, color='red', linestyle='--', label='Overbought')
    plt.axhline(30, color='green', linestyle='--', label='Oversold')
    plt.title('Relative Strength Index (RSI)')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid(True)
    plt.show();

def plot_macd(df):

    """Plot the Moving Average Convergence Divergence (MACD)."""

    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['MACD'], label='MACD', color='blue')
    plt.plot(df.index, df['MACD_signal'], label='MACD Signal', color='red')
    plt.bar(df.index, df['MACD_hist'], label='MACD Histogram', color='gray', alpha=0.3)
    plt.title('MACD and Signal Line')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()