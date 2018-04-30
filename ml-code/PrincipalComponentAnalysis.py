# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 15:33:57 2018

@author: averma
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

#sample data, 200 points
rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
plt.scatter(X[:, 0], X[:, 1])
plt.axis('equal');

#use PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)

#The fit learns some quantities from the data, most importantly the "components" and "explained variance":
print(pca.components_)
print(pca.explained_variance_)

# =============================================================================
# PCA as dimensionality reduction
# =============================================================================

pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)

#The transformed data has been reduced to a single dimension
#now plot this to visiulize it
X_new = pca.inverse_transform(X_pca)
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
plt.axis('equal');

# =============================================================================
# PCA for visualization: Hand-written digits
# =============================================================================
from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape #64 dimension

#use PCA now
pca = PCA(2)  # project from 64 to 2 dimensions
projected = pca.fit_transform(digits.data)
print(digits.data.shape)
print(projected.shape)

#We can now plot the first two principal components of each point to learn about the data:
plt.scatter(projected[:, 0], projected[:, 1],
            c=digits.target, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('spectral', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();

# =============================================================================
# Choosing the number of components
# =============================================================================

#very imp
pca = PCA().fit(digits.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

#we see that with the digits the first 10 components contain approximately 75% of the variance,
# while you need around 50 components to describe close to 100% of the variance.

#we'd need about 20 components to retain 90% of the variance

# =============================================================================
# PCA as Noise Filtering
# =============================================================================

def plot_digits(data):
    fig, axes = plt.subplots(4, 10, figsize=(10, 4),
                             subplot_kw={'xticks':[], 'yticks':[]},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8),
                  cmap='binary', interpolation='nearest',
                  clim=(0, 16))
plot_digits(digits.data)

#Now lets add some random noise to create a noisy dataset, and re-plot it:

np.random.seed(42)
noisy = np.random.normal(digits.data, 4)
plot_digits(noisy)

#Let's train a PCA on the noisy data, requesting that the projection preserve 50% of the variance:
pca = PCA(0.50).fit(noisy)
pca.n_components_ #= 12, so you will need 12 dimension to maintain 50% of variance

components = pca.transform(noisy)
filtered = pca.inverse_transform(components)
plot_digits(filtered) # it is much cleaner

# =============================================================================
# Example: Eigenfaces
# =============================================================================

from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)

#RandomizedPCA
#it contains a randomized method to approximate the first $N$ principal components 
#much more quickly than the standard PCA estimator, and thus is very useful 
#for high-dimensional data (here, a dimensionality of nearly 3,000)

#We will take a look at the first 150 components:
from sklearn.decomposition import RandomizedPCA
pca = RandomizedPCA(150)
pca.fit(faces.data)

# visualize the images
#these components are technically known as "eigenvectors," so these types of 
#images are often called "eigenfaces"). As you can see in this figure, 
#they are as creepy as they sound:

fig, axes = plt.subplots(3, 8, figsize=(9, 4),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape(62, 47), cmap='bone')
    
    
#asses how many doimensions we need
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

#We see that these 150 components account for just over 90% of the variance.

# Compute the components and projected faces
pca = RandomizedPCA(150).fit(faces.data)
components = pca.transform(faces.data)
projected = pca.inverse_transform(components)

# Plot the results
fig, ax = plt.subplots(2, 10, figsize=(10, 2.5),
                       subplot_kw={'xticks':[], 'yticks':[]},
                       gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i in range(10):
    ax[0, i].imshow(faces.data[i].reshape(62, 47), cmap='binary_r')
    ax[1, i].imshow(projected[i].reshape(62, 47), cmap='binary_r')
    
ax[0, 0].set_ylabel('full-dim\ninput')
ax[1, 0].set_ylabel('150-dim\nreconstruction');


#### check out, RandomizedPCA and SparsePCA, both are in the sklearn.decomposition submodule.
































