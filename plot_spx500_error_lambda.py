import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
from multiprocessing import Pool
from util import implyDensity

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

def scale(x):
    return x / 1000

def scaleInv(x):
    return x * 1000

tau = 0.082192
r = 0.0097
F = scale(2629.80)

def calculateSolutions(lambd, verbose=False):
    dfInput = pd.read_csv('SPX500.csv', delimiter=';', dtype={'Strike' : 'float', 'Volatility' : 'float'})
    strikesCall = strikesPut = [scale(strike) for strike in dfInput['Strike']]
    volatilities = dfInput['Volatility'] 

    from black import Black
    strikesPricesCallDict = {strike : Black.callPrice(F, strike, tau, vola, r) for (strike, vola) in zip(strikesCall, volatilities)}
    strikesPricesPutDict = {strike : Black.putPrice(F, strike, tau, vola, r) for (strike, vola) in zip(strikesPut, volatilities)}
    N = 1000
    xmin = 1.4
    xmax = 3.4
    nsv = 70

    S0 = F * np.exp(-r * tau)
    spots, phi, phiPrime, strikesCall, strikesPut, pricesCombined, pricesFit, chi2 = implyDensity(strikesPricesCallDict, strikesPricesPutDict, S0, r, tau, N, xmin, xmax, nsv, lambd, verbose=verbose)

    return lambd, chi2, spots, phi, pricesCombined, pricesFit, strikesCall

if __name__ == '__main__':
    filename = 'spx500.feather'
    if not os.path.exists(filename):
        lambdas = np.logspace(-10, 1, num=80)
        with Pool(16) as pool:
            results = pool.map(calculateSolutions, lambdas)
        errorsPrice = np.array([errorPrice for (_, errorPrice, _, _, _, _, _) in results])

        df = pd.DataFrame({'lambda' : lambdas, 'errorPrice': errorsPrice})
        df.to_feather(filename)
    else:
        df = pd.read_feather(filename)

    l1 = -7
    l2 = -4
    _, _, spots1, dens1, pricesCombined1, pricesFit1, strikesCall1 = calculateSolutions(10 ** l1, verbose=True)
    _, _, spots2, dens2, pricesCombined2, pricesFit2, strikesCall2 = calculateSolutions(10 ** l2, verbose=True)

    
    df = df.iloc[::3]

    fig, ax1 = plt.subplots(1, 1, figsize=(3.5, 2.0), sharex=True)

    ax1.axvline(l1, color='lightgrey') 
    ax1.axvline(l2, color='lightgrey')
    ax1.plot(np.log10(df['lambda']), np.log10(df['errorPrice']), '-o', color='k')
    ax1.set_ylabel(r'$\log_{10}( \chi^2 )$')
    ax1.set_ylim(-7, 2)
    ax1.xaxis.set_minor_locator(MultipleLocator(1))
    ax1.yaxis.set_minor_locator(MultipleLocator(1))
    ax1.yaxis.set_major_locator(MultipleLocator(2))
    ax1.set_xlabel(r'$\log_{10}(\lambda )$')

    plt.tight_layout()
    plt.savefig('errorsspx500.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()

    fig, ax3 = plt.subplots(1, 1, figsize=(3.5, 2), sharex=True, sharey=True)

    xr = 3.0
    xl = 1.8
    ax3.plot(spots1, dens1, color='k', label=r'$\lambda = 10^{' + '{}}}$'.format(l1))
    ax3.set_xlim(xl, xr)
    ax3.set_ylim(0, 6)
    ax3.set_ylabel(r'$\phi (x) \times 1000$')

    ax3.plot(spots2, dens2, color='r', label=r'$\lambda = 10^{' + '{}}}$'.format(l2), linestyle='--')
    ax3.legend()
    ax3.set_xlabel(r'$x / 1000$')
    ax3.xaxis.set_minor_locator(MultipleLocator(0.1))

    plt.tight_layout()
    plt.savefig('spx500density.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()

    from black import Black
    from scipy.optimize import bisect
    from scipy.interpolate import interp1d

    dfInput = pd.read_csv('SPX500.csv', delimiter=';', dtype={'Strike' : 'float', 'Volatility' : 'float'})
    strikes = [scale(strike) for strike in dfInput['Strike']]
    volatilities = dfInput['Volatility'] 

    def solveVolaCall(K, knownPrice):
        volaMin = 0.000001
        volaMax = 10000
        priceErrorFunc = lambda vola : Black.callPrice(F, K, tau, vola, r) - knownPrice
        x0 = bisect(priceErrorFunc, volaMin, volaMax)
        return x0

    strikesExtract = np.linspace(1.8, 3.05, num=3000)
    densityInterp = interp1d(spots1, dens1, kind='linear')
    spotsExtract = np.linspace(1.4, 3.4, num=10000)
    densityExtract = densityInterp(spotsExtract)
    pricesExtract = [np.trapz(np.multiply(np.maximum(0, spotsExtract - strike), densityExtract), spotsExtract) * np.exp(-r * tau) for strike in strikesExtract]
    volasPricesFit = [solveVolaCall(strike, knownPrice) for (strike, knownPrice) in zip(strikesExtract, pricesExtract)]
    
    plt.figure(figsize=(3.5, 3.5))
    plt.scatter(strikes, volatilities, label='original', marker='o', facecolors='none', color='k')
    plt.plot(strikesExtract, volasPricesFit, label='fit', color='r')
    plt.xlabel('$K / 1000$')
    plt.ylabel(r'$\sigma$')
    plt.xlim(1.8, 3.05)
    plt.ylim(0.15, 0.75)
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.2))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.legend()
    plt.tight_layout()
    plt.savefig('spx500volas.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()
