from AlgorithmImports import *
import tensorflow as tf
import numpy as np
import random
from transformers import TFBertForSequenceClassification, BertTokenizer, set_seed
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict
import torch
import torch.nn.functional as F
import scipy.stats as stats

from sentiment.sentiment_analysis import select_articles, get_sentiment_scores



def rebalance_portfolio(algorithm):
    weights = [0.1, 0.9]
    articles = select_articles(algorithm)
    sentiment_scores = get_sentiment_scores(algorithm, articles)
    scores = []
    directions = []
    option_scores = algorithm.get_option_model_outputs()

    for ticker in algorithm.tickers:
        pos_score = sentiment_scores.get(ticker, [0.33, 0.33])[0]
        neg_score = sentiment_scores.get(ticker, [0.33, 0.33])[1]
        option_score_pos = option_scores.get(ticker, [0.33, 0.33, 0.33])[1]
        option_score_neg = option_scores.get(ticker, [0.33, 0.33, 0.33])[0]
        score_pos = weights[0] * pos_score + weights[1] * option_score_pos
        score_neg = weights[0] * neg_score + weights[1] * option_score_neg
        score = max(score_pos, score_neg)
        flag = 1 if score_pos > score_neg else -1
        if score < algorithm.base_threshold:
            score = 0

        implied_volatility, VaR = get_implied_volatility_and_var(
            ticker, flag == 1, algorithm.history([ticker], 50, Resolution.DAILY)
        )
        if VaR > 0.05:
            score = 0
        score /= implied_volatility
        scores.append(score if score > algorithm.vol_threshold else 0)
        directions.append(flag)

    scores = scores if np.sum(scores) == 0 else scores / np.sum(scores)
    scores = np.multiply(np.array(scores), np.array(directions))

    for symbol, weight in zip(algorithm.tickers, scores):
        algorithm.set_holdings(symbol, weight * (1 - algorithm.cash_percentage))
