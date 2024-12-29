# region imports
from AlgorithmImports import *
# endregion


def get_implied_volatility_and_var(history_func, ticker, is_long, confidence_level=0.95):
    hist = history_func([ticker], 50, Resolution.DAILY)
    implied_volatility = 1
    var = 1
    if not hist.empty and "close" in hist.columns:
        implied_volatility = math.sqrt(265) * (hist["close"].pct_change(1) * 100).dropna().std()
        returns = hist['close'].pct_change().dropna()
        if returns.empty:
            return implied_volatility, var

        if is_long:
            var = -returns.quantile(1 - confidence_level)
        else:
            var = returns.quantile(confidence_level)
    return implied_volatility, var