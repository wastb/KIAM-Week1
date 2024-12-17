# Leveraging Financial News Sentiment for Stock Market Insights.

This project aims to use AI techniques to analyze financial news sentiment and predict stock prices. The project involves collecting financial news articles, analyzing their sentiment, and predicting stock movements based on the analyzed sentiment.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Nova Financial Solutions aims to enhance its predictive analytics capabilities to significantly boost its financial forecasting accuracy and operational efficiency through advanced data analysis. As a Data Analyst at Nova Financial Solutions,  your primary task is to conduct a rigorous analysis of the financial news dataset. The focus of your analysis should be two-fold:

- **Sentiment Analysis**: Perform sentiment analysis on the ‘headline’ text to quantify the tone and sentiment expressed in financial news. This will involve using natural language processing (NLP) techniques to derive sentiment scores, which can be associated with the respective 'Stock Symbol' to understand the emotional context surrounding stock-related news.

- **Correlation Analysis**: Establish statistical correlations between the sentiment derived from news articles and the corresponding stock price movements. This involves tracking stock price changes around the date the article was published and analyzing the impact of news sentiment on stock performance. This analysis should consider the publication date and potentially the time the article was published if such data can be inferred or is available.

## Installation

To set up this project, you will need to clone the repository and install the necessary dependencies. This project uses Python 3.11 and Conda for environment management.

1. Clone the repository:

    ```bash
    git clone https://github.com/wastb/KIAM-Week1
    ```

2. Create and activate a new environment:

    ```bash
    python -m venv 
   ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Sentiment Analysis**: Run the `sentiment_analyzer.py` script to perform sentiment analysis on the cleaned data.

    ```bash
    python scripts/sentiment_analyzer.py
    ```

2. **Exploratory Data Analysis (EDA)**: Use the `news_eda.py` and `stock_eda.py` script to conduct EDA on the dataset to understand the data distribution and key characteristics.

    ```bash
    python scripts/news_eda.py
    python scripts/stock_eda.py
    ```

3. **Notebooks**: The `notebooks/` directory contains Jupyter notebooks for Quantitative analysis, Sentiment Analysis and Correlation Analysis for each stock (e.g., `AAPL_quantitative_analysis.ipynb` for Apple). You can use these notebooks to experiment with different models and perform detailed analysis.

## Contributing

We welcome contributions to this project! If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

Please ensure your code adheres to the project's coding standards and is well-documented.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for more information.

