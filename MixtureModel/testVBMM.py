# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pylab as plt
from scipy.stats.distributions import norm
import VBMM

import json, matplotlib
s = json.load( open("../styles/bmh_matplotlibrc.json") )
matplotlib.rcParams.update(s)

def GenerateData(mu, sigma, p, N, plot=True):
    z = np.random.multinomial(1, p, N)
    ind = np.argmax(z, axis=1)
    x = [np.random.normal(mu[i], sigma[i]) for i in ind]

    if plot == True:
        plt.figure(1)
        plt.hist(x, bins=20)
        plt.show()       
    return np.asarray(x)
    
    
mu_true = [15, 30]
sigma_true = [1.5, 2]
p_true = [8/16., 8/16.]
N = 200

x = GenerateData(mu_true, sigma_true, p_true, N, plot=False)

vbmm =VBMM.VBMM(x, max_components=10)
vbmm.Fit()



mixtures = np.concatenate((vbmm.pis[vbmm.components],vbmm.means[vbmm.components],vbmm.sigmas[vbmm.components]))
mixtures = np.reshape(mixtures, (3,-1))

plt.hist(x, histtype='stepfilled', bins=50, alpha=0.85, color="#7A68A6", normed=True, label='Real data')
plt.xlabel('data')
plt.xlim(0, 60)
plt.ylim(0,0.25)

x_range = np.linspace(x.min()-1, x.max()+1, 500)
y_range = np.asarray([mixtures[0,i] * norm.pdf(x_range, mixtures[1,i],mixtures[2,i]) for i in range(mixtures.shape[1])])
y_range = np.sum(y_range, axis=0)
plt.plot(x_range, y_range, color="#A60628", linewidth=2, label='Esitmated pdf')

plt.legend()
plt.title('VBMM')
plt.show()