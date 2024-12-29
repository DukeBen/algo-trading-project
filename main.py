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

# code imports
from sentiment.keywords import keywords, tiingo_to_ticker
from algorithm.stop_loss import apply_stop_loss
from algorithm.var_and_volitility import get_implied_volatility_and_var
from algorithm.bsm_price import bsm_price
from models.ffnn import FFNN


'''
Note that due to the live subscription to Tiingo articles,
we are not able to abstract sentiment analysis code out of main
'''
class SentimentScores(QCAlgorithm):
    def initialize(self):
        #Intialize FinBERT for QC
        self.set_start_date(2017, 1, 1)
        self.set_end_date(2017, 2, 1)
        self.set_cash(10000000)
        self.cash_percentage = 0.5
        self.base_threshold = 0.5
        self.vol_threshold = 0.002

        self.prices = {}

        self.option_model = FFNN()
        self.option_model.load_state_dict(torch.load(self.object_store.get_file_path("group_1_model"), weights_only=True))
        
        self.max_vals = torch.Tensor([46.9949, 58.9286, 99.2321, 25.2211,  1.0000, 16.3904,  1.0000, 25.2013,
         1.5410, 18.3030,  9.2324,  0.5688, 73.1422, 73.1422])
        self.min_vals = torch.Tensor([ 2.2845e-01,  0.0000e+00,  0.0000e+00,  3.1372e-05, -8.0566e+03,
         3.6530e-05, -2.0593e+03, -2.3968e+00, -2.5187e+01, -1.6389e+01,
        -1.8807e+01, -1.6375e+01,  0.0000e+00,  0.0000e+00])
        
        set_seed(1, True)
        self.spy = self.add_equity('SPY', Resolution.Daily).symbol
        self.tickers = ["PFE", "KO", "DGX", "PKG", "TYL", "WLL", "MSFT", "HSBC", "UNH", "ASML", "TSM", "NEE", "BHP", "DIS", "WMT"]
        
        model_path = "ProsusAI/finbert"
        self._model = TFBertForSequenceClassification.from_pretrained(model_path, from_pt = True)
        self._tokenizer = BertTokenizer.from_pretrained(model_path)

        self.eq_symbols = {}
        self.option_symbols = {}
        self.ticker_to_tiingo = {}
        for ticker in self.tickers:
            eq = self.add_equity(ticker, Resolution.Daily)
            self.ticker_to_tiingo[ticker] = self.add_data(TiingoNews, ticker).symbol
            self.eq_symbols[ticker] = eq.symbol
            
            option = self.add_option(ticker)
            option.set_filter(timedelta(0), timedelta(30))
            self.option_symbols[ticker] = option.symbol

        self.max_num_articles = 10
        self.rebalance_period = 7
        self.tiingo_to_ticker = tiingo_to_ticker        
        self.keywords = keywords

        self.Schedule.On(self.DateRules.EveryDay(self.spy), self.TimeRules.Every(timedelta(hours=1)), lambda: apply_stop_loss(self))
        self.Schedule.On(self.date_rules.every(DayOfWeek.FRIDAY), self.TimeRules.AfterMarketClose(self.spy), self.rebalance_portfolio)
        self.countOfArticles = 0
        self.flag = 0


    def get_score(self, article_text):

        inputs = self._tokenizer(article_text, return_tensors='tf', max_length=512, truncation=True)
        outputs =self._model(**inputs)
        scores = tf.nn.softmax(outputs.logits, axis=-1).numpy()
        # positive, negative, neutral
        return scores

    def is_relevant(self, article_text, symbol):
        if str(symbol) not in self.keywords:
            return False
        tokens = word_tokenize(article_text.lower())
        return any(keyword in tokens for keyword in self.keywords[str(symbol)])
    

    

    def get_option_model_outputs(self):
        out = {}
        for ticker in self.tickers:
            symbol = self.option_symbols[ticker]
            chain = self.current_slice.option_chains.get(symbol)
            if chain:
                underlying = chain.underlying.price
                dt = self.time
                td = timedelta(days=14)
                desired_date = dt+td
                calls = [contract for contract in chain if contract.Right == OptionRight.Call]
                puts = [contract for contract in chain if contract.Right == OptionRight.Put]
                call_contracts = sorted(calls, key = lambda x: (abs(x.expiry - desired_date).days, abs(underlying - x.strike)))[:2]
                put_contracts = sorted(puts, key = lambda x: (abs(x.expiry - desired_date).days, abs(underlying - x.strike)))[:2]
                vol = (self.history([ticker], 15, Resolution.DAILY)["close"].pct_change(1)*100).dropna().std()
                if len(call_contracts) < 2 or len(put_contracts) < 2:
                    continue
                base_call = call_contracts[0]
                base_put = put_contracts[0]
                otm_call = call_contracts[1]
                otm_put = put_contracts[1]
                pcp = (base_call.ask_price-base_put.ask_price)/underlying
                skew = (otm_put.ask_price-base_call.ask_price)/underlying
                skew1 = (otm_call.ask_price-base_put.ask_price)/underlying
                skew2 = (otm_call.ask_price-base_call.ask_price)/underlying
                skew3 = (otm_put.ask_price-base_put.ask_price)/underlying
                skew4 = 100*(abs(otm_call.strike-underlying)-abs(base_call.strike-underlying))/underlying
                skew5 = 100*(abs(otm_put.strike-underlying)-abs(base_put.strike-underlying))/underlying
                flag = 1 if base_call.strike > otm_call.strike else 0
                bsm_0 = bsm_price(0, vol*np.sqrt(365)/100, underlying, call_contracts[0].strike)
                bsm_1 = bsm_price(1, vol*np.sqrt(365)/100, underlying, put_contracts[0].strike)
                call_bsm_diff = (base_call.ask_price-bsm_0)/base_call.ask_price
                put_bsm_diff = (base_put.ask_price-bsm_1)/base_put.ask_price
                vector = [vol, abs((base_call.expiry - desired_date).days)/14, 100*abs(otm_call.strike-underlying)/underlying, base_call.ask_price/underlying, call_bsm_diff, base_put.ask_price/underlying, put_bsm_diff, pcp, skew, skew1, skew2, skew3, skew4, skew5, flag]
                vector = torch.Tensor(vector)
                vector[:14] = (vector[:14]-self.min_vals)/(self.max_vals-self.min_vals)
                out[ticker]=self.option_model.forward(vector.reshape(1, 15)).reshape(3,).tolist()
            else:
                out[ticker]=[0.33, 0.33, 0.33]
        return out

        
    def add_tiingo_symbols(self):
        for ticker in self.tickers:
            self.ticker_to_tiingo[ticker] = self.add_data(TiingoNews, ticker).symbol
        self.debug(f'tiingo symbols readded')

    def remove_tiingo_symbols(self):
        for ticker in self.tickers:
            self.remove_security(self.ticker_to_tiingo[ticker])
        self.debug(f'tiingo symbols removed')

    def select_articles(self, been_called = False):
        if not been_called:
            return {}

        articles = {}
        for ticker in self.tickers:
            tiingo_symbol = self.ticker_to_tiingo[ticker]
            article_history = self.history(tiingo_symbol, self.rebalance_period, Resolution.DAILY)
            if len(article_history) > self.max_num_articles:
                article_history = article_history.sample(n=self.max_num_articles)
            articles[ticker] = article_history
        return articles

    def get_sentiment_scores(self, articles, ):
        #given the articles for each stock, find the sentiment scores for each stock
        sentiment_score = defaultdict(list)

        for ticker in self.ticker_to_tiingo.keys():
            article_history = articles[ticker]
            valid_articles = 0
            raw_sentiment = np.zeros(3)
            if len(article_history) == 0:
                self.debug(f'No article history for {ticker}')
                continue
            #total sentiment before dividing by articles
            for index, article in article_history.iterrows():
                if 'description' not in article:
                    continue
                article_description = article['description']
                if not self.is_relevant(article_description, self.ticker_to_tiingo[ticker]):
                    continue
                valid_articles += 1
                sentiment = self.get_score(article_description)
                raw_sentiment = np.add(sentiment, raw_sentiment)

            #divide by articles
            if not np.any(np.isnan(raw_sentiment)) and valid_articles > 0: #ensure no nan values
                sentiment_score[ticker] = raw_sentiment / valid_articles

        return sentiment_score

    def rebalance_portfolio(self):
        weights = [.1, .9]

        self.add_tiingo_symbols()
        articles = self.select_articles(True)
        sentiment_scores = self.get_sentiment_scores(articles)
        self.remove_tiingo_symbols()

        scores = []
        directions = []
        option_scores = self.get_option_model_outputs()
        for ticker in self.tickers:
            symbol = self.eq_symbols[ticker]

            pos_score = sentiment_scores[ticker][0][0] if ticker in sentiment_scores else 0.33
            neg_score = sentiment_scores[ticker][0][1] if ticker in sentiment_scores else 0.33

            option_score_pos = 0.33
            option_score_neg = 0.33
            if ticker in option_scores:
                option_score_pos = option_scores[ticker][1]
                option_score_neg = option_scores[ticker][0]
            score_pos = (weights[0]*pos_score+weights[1]*option_score_pos)
            score_neg = (weights[0]*neg_score+weights[1]*option_score_neg)
            score = max(score_pos, score_neg)
            flag = 1 if score_pos > score_neg else -1
            if score < self.base_threshold:
                score = 0
            
            implied_volatility, VaR = get_implied_volatility_and_var(self.history, ticker, flag == 1)
            if VaR > .05:
                score = 0
                self.debug(f'didnt invest in {ticker} because VaR was : {VaR}')
            score /= implied_volatility

            scores.append(score if score > self.vol_threshold else 0)
            directions.append(flag)

        scores = scores if np.sum(scores) == 0 else scores/np.sum(scores)
        scores = np.multiply(np.array(scores), np.array(directions))
    
        for symbol, weight in zip(self.tickers, scores):
            w = weight
            if weight < 0:
                w *= 0.8 # 20% margin
            self.set_holdings(symbol, w*(1-self.cash_percentage))
            if w == 0 and symbol in self.prices:
                del self.prices[symbol]
            elif w != 0:
                self.prices[symbol] = self.securities[symbol].price

        self.articles_by_symbol = {} #reset articles 
        self.articles_by_symbol_for_one_day = {}
