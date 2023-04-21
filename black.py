import math
from scipy.stats import norm

class Black:
    @staticmethod
    def _d1d2(F, K, T, sigma):
        d1 = (math.log(F / K) + 0.5 * sigma ** 2 * T) / sigma / math.sqrt(T)
        d2 = d1 - sigma * math.sqrt(T)
        return d1, d2

    @staticmethod
    def callPrice(F, K, T, sigma, r):
        d1, d2 = Black._d1d2(F, K, T, sigma)
        return math.exp(- r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))

    @staticmethod
    def putPrice(F, K, T, sigma, r):
        d1, d2 = Black._d1d2(F, K, T, sigma)
        return math.exp(- r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
