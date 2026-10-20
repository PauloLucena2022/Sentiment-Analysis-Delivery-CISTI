# Sentiment Analysis in Delivery Services using Ensemble Learning

Official repository for the research paper **"Sentiment Analysis in Delivery Services Using Ensemble Learning"**, presented at the **Iberian Conference on Information Systems and Technologies (CISTI)** and published by **Springer**.

**Paper Link:** [Springer Professional - Sentiment Analysis in Delivery Services Using Ensemble Learning](https://www.springerprofessional.de/en/sentiment-analysis-in-delivery-services-using-ensemble-learning-/53465138)

## Project Overview

This project investigaes user sentiment in Brazilian food delivery applications by analyzing 35,982 real customer reviews. 

The pipeline encompasses data ingestion, data quality verification, statistical text analysis, exploratory data analysis (EDA), natural language processing (NLP) and sentiment classification using Ensemble Learning. 

## Key Exploratory Data Analysis (EDA) Insights

* **Dataset Volume:** 35,982 user reviews across 14 features, with zero missing values in critical fields (`content` and `score`).
* **Review Length Distribution:** Reviews contain an average of **28.09 words** (ranging from 1 to 350 words), with 2,255 single-word reviews (e.g., *"Ótimo"*, *"Lixo"*, *"Eficiente"*).
* **Target Class Distribution (`score` 1 to 5):**
  * **Score 5:** 9,000 instances (25.01%)
  * **Score 4:** 9,000 instances (25.01%)
  * **Score 3:** 5,994 instances (16.66%)
  * **Score 2:** 5,994 instances (16.66%)
  * **Score 1:** 5,994 instances (16.66%)

## Repository Architecture

```text
sentiment-analysis-delivery/
├── data/
│   ├── raw/                   # Immutable original dataset (delivery_alimentos.csv)
│   └── processed/             # Cleaned data and engineered features
├── models/                    # Serialized models and vectorizers (.pkl)
├── notebooks/
│   └── Sentiment_Analysis_Delivery.ipynb  # Original experimental notebook
├── reports/
│   └── figures/               # Automatically generated plots (.png)
├── src/
│   ├── __init__.py
│   ├── data/
│   │   └── load_data.py       # Data ingestion and null-check routines
│   ├── features/
│   │   ├── build_features.py  # Text cleaning, stopwords, stemming, and feature engineering
│   │   └── text_stats.py      # Word count metrics and N-gram/frequency extraction
│   ├── models/
│   │   ├── evaluate_model.py  # Evaluation metrics (Accuracy, Classification Report)
│   │   └── train_model.py     # Data splitting, vectorization, and Ensemble training
│   └── visualization/
│       └── plots.py           # Automated plotting functions (histograms, pie charts, bar plots)
├── main.py                    # Main orchestrator script for the Machine Learning pipeline
├── run_eda.py                 # Orchestrator script for Exploratory Data Analysis (EDA)
├── .gitignore                 # Rules to ignore virtual environments and large data files
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```
