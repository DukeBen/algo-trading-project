# region imports
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
# endregion

def bsm_price(option_type, sigma, s, k, T=14/365, r=0.032, q=0.01):
        # calculate the bsm price of European call and put options
        d1 = (np.log(s / k) + (r - q + sigma ** 2 * 0.5) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 0:
            return np.exp(-r*T) * (s * np.exp((r - q)*T) * stats.norm.cdf(d1) - k *  stats.norm.cdf(d2))
        return np.exp(-r*T) * (k * stats.norm.cdf(-d2) - s * np.exp((r - q)*T) *  stats.norm.cdf(-d1))