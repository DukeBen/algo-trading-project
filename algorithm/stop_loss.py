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


def apply_stop_loss(algorithm):
    portfolio_value = algorithm.Portfolio.TotalPortfolioValue
    for ticker in algorithm.tickers:
        if algorithm.portfolio[ticker].invested and ticker in algorithm.prices:
            if algorithm.portfolio[ticker].is_long and algorithm.securities[ticker].price > algorithm.prices[ticker]:
                algorithm.prices[ticker] = algorithm.securities[ticker].price
            if algorithm.portfolio[ticker].is_short and algorithm.securities[ticker].price < algorithm.prices[ticker]:
                algorithm.prices[ticker] = algorithm.securities[ticker].price
        position = algorithm.Portfolio[ticker]
        if position.Invested:
            current_price = algorithm.Securities[ticker].Price
            position_direction = 1 if position.is_long else -1
            high_price = algorithm.prices[ticker] if ticker in algorithm.prices else position.average_price
            curr_loss = position_direction * (high_price - current_price) / high_price
            if curr_loss > 0.1:
                algorithm.liquidate(ticker)
                if ticker in algorithm.prices:
                    del algorithm.prices[ticker]
