import numpy as np
import cvxpy as cp
from scipy.sparse.linalg import svds

def _getKernel(spots, deltaX, strikesCall, strikesPut, r, tau):
    M = len(strikesCall) + len(strikesPut)
    N = len(spots)
    kernel = np.zeros((M, N))
    kernel[0:len(strikesPut), :] = np.maximum(0, np.subtract.outer(strikesPut, spots))
    kernel[len(strikesPut):M, :] = np.maximum(0, np.subtract.outer(spots, strikesCall).T)
    kernel[:, 0] *= 0.5
    kernel[:, N-1] *= 0.5
    kernel *= deltaX * np.exp(-r * tau)
    return kernel

def _getSVD(matrix, nsv=None):
    if nsv == None or nsv >= min(matrix.shape) or nsv <= 0:
        raise Exception("nsv must be 0 < nsv < {} (min(matrix.shape))".format(min(matrix.shape)))
    U, S, V = svds(matrix, k=nsv, which='LM')
    S = np.flip(S, axis=0)
    U = np.flip(U, axis=1)
    Vt = np.flip(V, axis=0)
    return U, S, Vt

def _solveProblem(deltaX, U, S, Vt, prices, lambd, verbose=True):
    Q = len(S)
    Smat = np.diag(S)
    phiPrime = cp.Variable(Q)
    errTerm = prices - U @ Smat @ phiPrime
    objective = cp.Minimize(0.5 * cp.norm2(errTerm) ** 2 + lambd * cp.norm1(phiPrime))
    VMod = np.copy(Vt.T)
    VMod[0, :] *= 0.5
    VMod[-1, :] *= 0.5
    constraints = [Vt.T @ phiPrime >= 0, cp.sum(deltaX * VMod @ phiPrime) == 1]
    prob = cp.Problem(objective, constraints)
    _ = prob.solve(verbose=verbose, solver='ECOS')
    pricesFit = np.dot(U, np.dot(Smat, phiPrime.value))
    chi2 = 0.5 * np.linalg.norm(prices - pricesFit, ord=2) ** 2
    return phiPrime.value, pricesFit, chi2

def _getForwardPrice(S0, r, tau):
    return S0 * np.exp(r * tau)

def _getOTMPricesCall(strikesPricesCall, S0, r, tau):
    forwardPrice = _getForwardPrice(S0, r, tau) * 0.8
    resCall = [(strike, priceCall) for (strike, priceCall) in strikesPricesCall.items() if strike >= forwardPrice]
    resPut = [(strike, priceCall - S0 + strike * np.exp(-r * tau)) for (strike, priceCall) in strikesPricesCall.items() if strike < forwardPrice]
    #return resCall, resPut
    return list(strikesPricesCall.items()), []

def _getOTMPricesPut(strikesPricesPut, S0, r, tau):
    forwardPrice = _getForwardPrice(S0, r, tau) * 1.2
    resPut = [(strike, pricePut) for (strike, pricePut) in strikesPricesPut.items() if strike <= forwardPrice]
    resCall = [(strike, pricePut + S0 - strike * np.exp(-r * tau)) for (strike, pricePut) in strikesPricesPut.items() if strike > forwardPrice]
    #return resCall, resPut
    return [], list(strikesPricesPut.items())

def _getOTMPrices(strikesPricesCall, strikesPricesPut, S0, r, tau):
    resCall1, resPut1 = _getOTMPricesCall(strikesPricesCall, S0, r, tau)
    resCall2, resPut2 = _getOTMPricesPut(strikesPricesPut, S0, r, tau)
    resCall = sorted(resCall1 + resCall2, key=lambda tup : tup[0])
    resPut = sorted(resPut1 + resPut2, key=lambda tup : tup[0])
    strikesCall = np.array([strike for (strike, _) in resCall])
    strikesPut = np.array([strike for (strike, _) in resPut])
    pricesCall = np.array([price for (_, price) in resCall])
    pricesPut = np.array([price for (_, price) in resPut])
    return strikesCall, pricesCall, strikesPut, pricesPut

def implyDensity(strikesPricesCallDict, strikesPricesPutDict, S0, r, tau, N, xmin, xmax, nsv, lambd, verbose=True):
    assert lambd >= 0
    strikesCall, pricesCall, strikesPut, pricesPut = _getOTMPrices(strikesPricesCallDict, strikesPricesPutDict, S0, r, tau)
    assert xmin < xmax
    spots = np.linspace(xmin, xmax, num=N, endpoint=True)
    assert N > 0
    deltaX = spots[1] - spots[0]
    kernel = _getKernel(spots, deltaX, strikesCall, strikesPut, r, tau)
    U, S, Vt = _getSVD(kernel, nsv=nsv)
    pricesCombined = np.concatenate((pricesPut, pricesCall), axis=None)
    phiPrime, pricesFit, chi2 = _solveProblem(deltaX, U, S, Vt, pricesCombined, lambd, verbose=verbose)
    phi = np.dot(Vt.T, phiPrime)
    return spots, phi, phiPrime, strikesCall, strikesPut, pricesCombined, pricesFit, chi2
