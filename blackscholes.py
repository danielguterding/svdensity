import math
from scipy.stats import norm

class BlackScholes:
    @staticmethod
    def _d1d2(S, K, t, T, sigma, r):
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * (T - t)) / sigma / math.sqrt(T - t)
        d2 = d1 - sigma * math.sqrt(T - t)
        return d1, d2

    @staticmethod
    def callPrice(S, K, t, T, sigma, r):
        d1, d2 = BlackScholes._d1d2(S, K, t, T, sigma, r)
        return S * norm.cdf(d1) - K * math.exp(-r * (T - t)) * norm.cdf(d2)

    @staticmethod
    def putPrice(S, K, t, T, sigma, r):
        d1, d2 = BlackScholes._d1d2(S, K, t, T, sigma, r)
        return K * math.exp(-r * (T - t)) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def vega(S, K, t, T, sigma, r):
        _, d2 = BlackScholes._d1d2(S, K, t, T, sigma, r)
        return K * math.exp(-r * (T - t)) * norm.pdf(d2) * math.sqrt(T -t)
