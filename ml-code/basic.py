# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 17:17:19 2018

@author: averma
"""

# =============================================================================
# Data Representation in Scikit-Learn
# =============================================================================

#Data as table

import seaborn as sns
iris = sns.load_dataset('iris')
iris.head()

#Features matrix

sns.set()
sns.pairplot(iris, hue='species', size=1.5);

#extract the features matrix and target array from the DataFrame
X_iris = iris.drop('species', axis=1)
X_iris.shape

y_iris = iris['species']
y_iris.shape

# =============================================================================
# Scikit-Learn's Estimator API
# =============================================================================

# =============================================================================
# Supervised learning example: Simple linear regression
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)
plt.scatter(x, y);

#1. Choose a class of model
from sklearn.linear_model import LinearRegression

#2. Choose model hyperparameters
model = LinearRegression(fit_intercept=True)
model

#3. Arrange data into a features matrix and target vector
X = x[:, np.newaxis]
X.shape

#4. Fit the model to your data
model.fit(X, y)

#test if the coefficinet and intercept is coming properly
model.coef_
model.intercept_

#5. Predict labels for unknown data
xfit = np.linspace(-1, 11)
Xfit = xfit[:, np.newaxis] # this to convert (50,) to (50,1) shape
yfit = model.predict(Xfit)
#Finally, let's visualize the results by plotting first the raw data, and then this model fit
plt.scatter(x, y)
plt.plot(xfit, yfit);

# =============================================================================
# Supervised learning example: Iris classification
# =============================================================================
#1. split the data into a training set and a testing set.
from sklearn.cross_validation import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris,random_state=1)

from sklearn.naive_bayes import GaussianNB # 1. choose model class
model = GaussianNB()                       # 2. instantiate model
model.fit(Xtrain, ytrain)                  # 3. fit model to data
y_model = model.predict(Xtest)             # 4. predict on new data

#2. Finally, we can use the accuracy_score utility to see the fraction of predicted labels that match their true value
from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)

# =============================================================================
# Unsupervised learning example: Iris dimensionality
# =============================================================================

from sklearn.decomposition import PCA  # 1. Choose the model class
model = PCA(n_components=2)            # 2. Instantiate the model with hyperparameters
model.fit(X_iris)                      # 3. Fit to data. Notice y is not specified!
X_2D = model.transform(X_iris)         # 4. Transform the data to two dimensions

#Now let's plot the results. A quick way to do this is to insert the results 
#into the original Iris DataFrame, and use Seaborn's lmplot to show the results:

iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]
sns.lmplot("PCA1", "PCA2", hue='species', data=iris, fit_reg=False);

# =============================================================================
# Unsupervised learning: Iris clustering
# =============================================================================

from sklearn.mixture import GMM      # 1. Choose the model class
model = GMM(n_components=3,
            covariance_type='full')  # 2. Instantiate the model with hyperparameters
model.fit(X_iris)                    # 3. Fit to data. Notice y is not specified!
y_gmm = model.predict(X_iris)        # 4. Determine cluster labels

iris['cluster'] = y_gmm
sns.lmplot("PCA1", "PCA2", data=iris, hue='species',
           col='cluster', fit_reg=False);


# =============================================================================
# Application: Exploring Hand-written Digits
# =============================================================================

#Loading and visualizing the digits data

from sklearn.datasets import load_digits
digits = load_digits()
digits.images.shape

#Let's visualize the first hundred of these

import matplotlib.pyplot as plt

fig, axes = plt.subplots(10, 10, figsize=(8, 8),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(digits.target[i]),
            transform=ax.transAxes, color='green')

#built the data and target attributes

X = digits.data
X.shape

y = digits.target
y.shape

# =============================================================================
# Unsupervised learning: Dimensionality reduction
# =============================================================================
#drag digits from 64 dimensions to 2
from sklearn.manifold import Isomap
iso = Isomap(n_components=2)
iso.fit(digits.data)
data_projected = iso.transform(digits.data)
data_projected.shape # this will give 2d array

#plot this
plt.scatter(data_projected[:, 0], data_projected[:, 1], c=digits.target,
            edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('spectral', 10))
plt.colorbar(label='digit label', ticks=range(10))
plt.clim(-0.5, 9.5);

#Classification on digits
#Let's apply a classification algorithm to the digits

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest) # cretae model, predict

#gauge its accuracy by comparing the true values of the test set to the predictions
from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)

#this single number doesn't tell us where we've gone wrong—one nice way 
#to do this is to use the confusion matrix, which we can compute with Scikit-Learn and plot with Seaborn

from sklearn.metrics import confusion_matrix

mat = confusion_matrix(ytest, y_model)

sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value');


#We'll use green for correct labels, and red for incorrect labels:

fig, axes = plt.subplots(10, 10, figsize=(8, 8),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

test_images = Xtest.reshape(-1, 8, 8)

for i, ax in enumerate(axes.flat):
    ax.imshow(test_images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(y_model[i]),
            transform=ax.transAxes,
            color='green' if (ytest[i] == y_model[i]) else 'red')
    

