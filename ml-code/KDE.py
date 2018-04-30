# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 16:49:31 2018

@author: averma
"""

# =============================================================================
# Kernel Density Estimation
# =============================================================================

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

# =============================================================================
# Motivating KDE: Histograms
# =============================================================================
def make_data(N, f=0.3, rseed=1):
    rand = np.random.RandomState(rseed)
    x = rand.randn(N)
    x[int(f * N):] += 5
    return x

x = make_data(1000)

hist = plt.hist(x, bins=30, normed=True)

#histogram are not reliable
x = make_data(20)
bins = np.linspace(-5, 10, 10)

fig, ax = plt.subplots(1, 2, figsize=(12, 4),
                       sharex=True, sharey=True,
                       subplot_kw={'xlim':(-4, 9),
                                   'ylim':(-0.02, 0.3)})
fig.subplots_adjust(wspace=0.05)
for i, offset in enumerate([0.0, 0.6]):
    ax[i].hist(x, bins=bins + offset, normed=True)
    ax[i].plot(x, np.full_like(x, -0.01), '|k',
               markeredgewidth=1)
    
# you would probably not guess that these two histograms were built from the same data: 

#we can think of a histogram as a stack of blocks
fig, ax = plt.subplots()
bins = np.arange(-3, 8)
ax.plot(x, np.full_like(x, -0.1), '|k',
        markeredgewidth=1)
for count, edge in zip(*np.histogram(x, bins)):
    for i in range(count):
        ax.add_patch(plt.Rectangle((edge, i), 1, 1,
                                   alpha=0.5))
ax.set_xlim(-4, 8)
ax.set_ylim(-0.2, 8)

#instead of stacking the blocks aligned with the bins, 
#we were to stack the blocks aligned with the points they represent
x_d = np.linspace(-4, 8, 2000)
density = sum((abs(xi - x_d) < 0.5) for xi in x)

plt.fill_between(x_d, density, alpha=0.5)
plt.plot(x, np.full_like(x, -0.1), '|k', markeredgewidth=1)

plt.axis([-4, 8, -0.2, 8]);

#replace the blocks at each location with a smooth function, like a Gaussian.
# Let's use a standard normal curve at each point instead of a block:

from scipy.stats import norm
x_d = np.linspace(-4, 8, 1000)
density = sum(norm(xi).pdf(x_d) for xi in x)

plt.fill_between(x_d, density, alpha=0.5)
plt.plot(x, np.full_like(x, -0.1), '|k', markeredgewidth=1)

plt.axis([-4, 8, -0.2, 5]);


# =============================================================================
# Kernel Density Estimation in Practice
# =============================================================================

from sklearn.neighbors import KernelDensity

# instantiate and fit the KDE model
kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
kde.fit(x[:, None])

# score_samples returns the log of the probability density
logprob = kde.score_samples(x_d[:, None])

plt.fill_between(x_d, np.exp(logprob), alpha=0.5)
plt.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
plt.ylim(-0.02, 0.22)

# =============================================================================
# Selecting the bandwidth via cross-validation
# =============================================================================

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import LeaveOneOut

bandwidths = 10 ** np.linspace(-1, 1, 100)
grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                    {'bandwidth': bandwidths},
                    cv=LeaveOneOut(len(x)))
grid.fit(x[:, None]);

#Now we can find the choice of bandwidth which maximizes the score 
#(which in this case defaults to the log-likelihood):

grid.best_params_


# =============================================================================
# Example: Not-So-Naive Bayes
# =============================================================================
#https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html#Example:-Not-So-Naive-Bayes
#TODO:

