# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:02:43 2018

@author: averma
"""

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
#Here is a function that will create data in the shape of the word "HELLO":

def make_hello(N=1000, rseed=42):
    # Make a plot with "HELLO" text; save as PNG
    fig, ax = plt.subplots(figsize=(4, 1))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    ax.text(0.5, 0.4, 'HELLO', va='center', ha='center', weight='bold', size=85)
    fig.savefig('hello.png')
    plt.close(fig)
    
    # Open this PNG and draw random points from it
    from matplotlib.image import imread
    data = imread('hello.png')[::-1, :, 0].T
    rng = np.random.RandomState(rseed)
    X = rng.rand(4 * N, 2)
    i, j = (X * data.shape).astype(int).T
    mask = (data[i, j] < 1)
    X = X[mask]
    X[:, 0] *= (data.shape[0] / data.shape[1])
    X = X[:N]
    return X[np.argsort(X[:, 0])]

#Let's call the function and visualize the resulting data:
X = make_hello(1000)
colorize = dict(c=X[:, 0], cmap=plt.cm.get_cmap('rainbow', 5))
plt.scatter(X[:, 0], X[:, 1], **colorize)
plt.axis('equal');
   
# =============================================================================
# Multidimensional Scaling (MDS)
# =============================================================================

#use a rotation matrix to rotate the data, the x and y values change, 
#but the data is still fundamentally the same
def rotate(X, angle):
    theta = np.deg2rad(angle)
    R = [[np.cos(theta), np.sin(theta)],
         [-np.sin(theta), np.cos(theta)]]
    return np.dot(X, R)
    
X2 = rotate(X, 20) + 5
plt.scatter(X2[:, 0], X2[:, 1], **colorize)
plt.axis('equal');

#we construct an n*n array such that entry $(i, j)$ contains the distance 
#between point $i$ and point $j$. Let's use Scikit-Learn's efficient pairwise_distances 
#function to do this for our original data:

from sklearn.metrics import pairwise_distances
D = pairwise_distances(X)
D.shape

#for our N=1,000 points, we obtain a 1000×1000 matrix, which can be visualized as shown here
plt.imshow(D, zorder=2, cmap='Blues', interpolation='nearest')
plt.colorbar();

#similarly construct a distance matrix for our rotated and translated data, 
#we see that it is the same:

#The MDS algorithm recovers one of the possible two-dimensional coordinate 
#representations of our data, using only the N*N distance matrix describing the 
#relationship between the data points.

from sklearn.manifold import MDS
model = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
out = model.fit_transform(D)
plt.scatter(out[:, 0], out[:, 1], **colorize)
plt.axis('equal');

# =============================================================================
# MDS as Manifold Learning
# =============================================================================

def random_projection(X, dimension=3, rseed=42):
    assert dimension >= X.shape[1]
    rng = np.random.RandomState(rseed)
    C = rng.randn(dimension, dimension)
    e, V = np.linalg.eigh(np.dot(C, C.T))
    return np.dot(X, V[:X.shape[1]])
    
X3 = random_projection(X, 3)
X3.shape

#lets visualize these points to see what we're working with:

from mpl_toolkits import mplot3d
ax = plt.axes(projection='3d')
ax.scatter3D(X3[:, 0], X3[:, 1], X3[:, 2],**colorize)
ax.view_init(azim=70, elev=50)

#We can now ask the MDS estimator to input this three-dimensional data, 
#compute the distance matrix, and then determine the optimal 
#two-dimensional embedding for this distance matrix. 
#The result recovers a representation of the original data:

model = MDS(n_components=2, random_state=1)
out3 = model.fit_transform(X3)
plt.scatter(out3[:, 0], out3[:, 1], **colorize)
plt.axis('equal');

# =============================================================================
# Nonlinear Embeddings: Where MDS Fails
# =============================================================================
#Consider the following embedding, which takes the input and 
#contorts it into an "S" shape in three dimensions:

def make_hello_s_curve(X):
    t = (X[:, 0] - 2) * 0.75 * np.pi
    x = np.sin(t)
    y = X[:, 1]
    z = np.sign(t) * (np.cos(t) - 1)
    return np.vstack((x, y, z)).T

XS = make_hello_s_curve(X)

#This is again three-dimensional data, but we can see that the
# embedding is much more complicated
from mpl_toolkits import mplot3d
ax = plt.axes(projection='3d')
ax.scatter3D(XS[:, 0], XS[:, 1], XS[:, 2],**colorize);

#If we try a simple MDS algorithm on this data, it is not able to 
#"unwrap" this nonlinear embedding, and we lose track of the fundamental 
#relationships in the embedded manifold:

from sklearn.manifold import MDS
model = MDS(n_components=2, random_state=2)
outS = model.fit_transform(XS)
plt.scatter(outS[:, 0], outS[:, 1], **colorize)
plt.axis('equal');

# =============================================================================
# Nonlinear Manifolds: Locally Linear Embedding
# =============================================================================
#locally linear embedding (LLE): rather than preserving all distances, 
#it instead tries to preserve only the distances between neighboring points

from sklearn.manifold import LocallyLinearEmbedding
model = LocallyLinearEmbedding(n_neighbors=100, n_components=2, method='modified',
                               eigen_solver='dense')
out = model.fit_transform(XS)

fig, ax = plt.subplots()
ax.scatter(out[:, 0], out[:, 1], **colorize)
ax.set_ylim(0.15, -0.15);

# =============================================================================
# Example: Isomap on Faces
# =============================================================================
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=30)
faces.data.shape

#Let's quickly visualize several of these images to see what we're working with
fig, ax = plt.subplots(4, 8, subplot_kw=dict(xticks=[], yticks=[]))
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='gray')
    
#One useful way to start is to compute a PCA, and examine the 
#explained variance ratio, which will give us an idea of how many 
#linear features are required to describe the data:
from sklearn.decomposition import RandomizedPCA
model = RandomizedPCA(100).fit(faces.data)
plt.plot(np.cumsum(model.explained_variance_ratio_))
plt.xlabel('n components')
plt.ylabel('cumulative variance');

#When this is the case, nonlinear manifold embeddings like LLE and 
#Isomap can be helpful. We can compute an Isomap embedding on these 
#faces using the same pattern shown before
from sklearn.manifold import Isomap
model = Isomap(n_components=2)
proj = model.fit_transform(faces.data)
proj.shape

#output is a two-dimensional projection of all the input images. 
#To get a better idea of what the projection tells us,
#let's define a function that will output image thumbnails at the 
#locations of the projections

from matplotlib import offsetbox

def plot_components(data, model, images=None, ax=None,
                    thumb_frac=0.05, cmap='gray'):
    ax = ax or plt.gca()
    
    proj = model.fit_transform(data)
    ax.plot(proj[:, 0], proj[:, 1], '.k')
    
    if images is not None:
        min_dist_2 = (thumb_frac * max(proj.max(0) - proj.min(0))) ** 2
        shown_images = np.array([2 * proj.max(0)])
        for i in range(data.shape[0]):
            dist = np.sum((proj[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                # don't show points that are too close
                continue
            shown_images = np.vstack([shown_images, proj[i]])
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i], cmap=cmap),
                                      proj[i])
            ax.add_artist(imagebox)

#Calling this function now, we see the result:
fig, ax = plt.subplots(figsize=(10, 10))
plot_components(faces.data,
                model=Isomap(n_components=2),
                images=faces.images[:, ::2, ::2])



