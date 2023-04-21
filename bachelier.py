import math
from scipy.stats import norm

class Bachelier:
    @staticmethod
    def _forwardPrice(S, t, T, r):
        return S * math.exp(r * (T - t))

    @staticmethod
    def _moneyness(F, K, t, T, sigma):
        return (F - K) / sigma / math.sqrt(T - t)

    @staticmethod
    def callPriceForward(F, K, t, T, sigma, r):
        m = Bachelier._moneyness(F, K, t, T, sigma)
        result = (F - K) * norm.cdf(m) + sigma * math.sqrt(T - t) * norm.pdf(m)
        return math.exp(-r * (T - t)) * result

    @staticmethod
    def callPrice(S, K, t, T, sigma, r):
        F = Bachelier._forwardPrice(S, t, T, r)
        return Bachelier.callPriceForward(F, K, t, T, sigma, r)

    @staticmethod
    def putPriceForward(F, K, t, T, sigma, r):
        m = Bachelier._moneyness(F, K, t, T, sigma)
        result = (K - F) * norm.cdf(-m) + sigma * math.sqrt(T - t) * norm.pdf(m)
        return math.exp(-r * (T - t)) * result

    @staticmethod
    def putPrice(S, K, t, T, sigma, r):
        F = Bachelier._forwardPrice(S, t, T, r)
        return Bachelier.putPriceForward(F, K, t, T, sigma, r)
