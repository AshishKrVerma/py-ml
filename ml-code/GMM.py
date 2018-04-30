# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 14:08:00 2018

@author: averma
"""
# =============================================================================
# Gaussian Mixture Models
# =============================================================================

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

# =============================================================================
# Motivating GMM: Weaknesses of k-Means
# =============================================================================

# Generate some data
from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=400, centers=4,
                       cluster_std=0.60, random_state=0)
X = X[:, ::-1] # flip axes for better plotting

# Plot the data with K Means Labels
from sklearn.cluster import KMeans
kmeans = KMeans(4, random_state=0)
labels = kmeans.fit(X).predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');

#One way to think about the k-means model is that it places a circle 
#(or, in higher dimensions, a hyper-sphere) at the center of each cluster,

from scipy.spatial.distance import cdist

def plot_kmeans(kmeans, X, n_clusters=4, rseed=0, ax=None):
    labels = kmeans.fit_predict(X)

    # plot the input data
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)

    # plot the representation of the KMeans model
    centers = kmeans.cluster_centers_
    radii = [cdist(X[labels == i], [center]).max()
             for i, center in enumerate(centers)]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))

#now print the circles

kmeans = KMeans(n_clusters=4, random_state=0)
plot_kmeans(kmeans, X)

#k-means has no built-in way of accounting for oblong or elliptical clusters.
# So, for example, if we take the same data and transform it, 
#the cluster assignments end up becoming muddled:
rng = np.random.RandomState(13)
X_stretched = np.dot(X, rng.randn(2, 2))

kmeans = KMeans(n_clusters=4, random_state=0)
plot_kmeans(kmeans, X_stretched)

# =============================================================================
# Generalizing E–M: Gaussian Mixture Models
# =============================================================================
from sklearn.mixture import GMM
gmm = GMM(n_components=4).fit(X)
labels = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');

probs = gmm.predict_proba(X)
print(probs[:5].round(3))

size = 50 * probs.max(1) ** 2  # square emphasizes differences
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=size);

#Let's create a function that will help us visualize the locations and shapes 
#of the GMM clusters by drawing ellipses based on the GMM output
from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covars_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

#With this in place, we can take a look at what the four-component 
#GMM gives us for our initial data:
gmm = GMM(n_components=4, random_state=42)
plot_gmm(gmm, X)
#fit our stretched dataset
gmm = GMM(n_components=4, covariance_type='full', random_state=42)
plot_gmm(gmm, X_stretched)

# =============================================================================
# Choosing the covariance type
# =============================================================================
#The default is covariance_type="diag"
#covariance_type="spherical"
#covariance_type="full"

# =============================================================================
# GMM as Density Estimation
# =============================================================================

from sklearn.datasets import make_moons
Xmoon, ymoon = make_moons(200, noise=.05, random_state=0)
plt.scatter(Xmoon[:, 0], Xmoon[:, 1]);

gmm2 = GMM(n_components=2, covariance_type='full', random_state=0)
plot_gmm(gmm2, Xmoon)

#But if we instead use many more components and ignore the cluster labels, 
#we find a fit that is much closer to the input data
gmm16 = GMM(n_components=16, covariance_type='full', random_state=0)
plot_gmm(gmm16, Xmoon, label=False)

#GMM gives us the recipe to generate new random data distributed similarly to our input
#For example, here are 400 new points drawn from this 16-component GMM fit to our original data:
Xnew = gmm16.sample(400, random_state=42)
plt.scatter(Xnew[:, 0], Xnew[:, 1]);

# =============================================================================
# How many components
# =============================================================================

#Another means of correcting for over-fitting is to adjust the model likelihoods 
#using some analytic criterion such as the Akaike information criterion (AIC) or 
#the Bayesian information criterion (BIC). Scikit-Learn's GMM estimator actually 
#includes built-in methods that compute both of these, and so it is very easy to 
#operate on this approach.

n_components = np.arange(1, 21)
models = [GMM(n, covariance_type='full', random_state=0).fit(Xmoon)
          for n in n_components]

plt.plot(n_components, [m.bic(Xmoon) for m in models], label='BIC')
plt.plot(n_components, [m.aic(Xmoon) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components');

#The AIC tells us that our choice of 16 components above was probably too many: 
#around 8-12 components would have been a better choice. As is typical with 
#this sort of problem, the BIC recommends a simpler model.

# =============================================================================
# Example: GMM for Generating New Data
# =============================================================================
from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape


#Next let's plot the first 100 of these to recall exactly what we're looking at
def plot_digits(data):
    fig, ax = plt.subplots(10, 10, figsize=(8, 8),
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8, 8), cmap='binary')
        im.set_clim(0, 16)
plot_digits(digits.data)

#Here we will use a straightforward PCA, asking it to preserve 99% of the variance in the projected data
from sklearn.decomposition import PCA
pca = PCA(0.99, whiten=True)
data = pca.fit_transform(digits.data)
data.shape #41 dimensions now

#let's use the AIC to get a gauge for the number of GMM components we should use:
n_components = np.arange(50, 210, 10)
models = [GMM(n, covariance_type='full', random_state=0)
          for n in n_components]
aics = [model.fit(data).aic(data) for model in models]
plt.plot(n_components, aics);

#It appears that around 110 components minimizes the AIC; we will use this model. 
#Let's quickly fit this to the data and confirm that it has converged:

gmm = GMM(110, covariance_type='full', random_state=0)
gmm.fit(data)
print(gmm.converged_)

#Now we can draw samples of 100 new points within this 41-dimensional projected space,
# using the GMM as a generative model:

data_new = gmm.sample(100, random_state=0)
data_new.shape

#Finally, we can use the inverse transform of the PCA object to construct the new digits.

digits_new = pca.inverse_transform(data_new)
plot_digits(digits_new)

#The results for the most part look like plausible digits from the dataset !!!

