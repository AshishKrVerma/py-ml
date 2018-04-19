# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 11:11:54 2018

@author: averma
"""

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

# =============================================================================
# Holdout sets
# =============================================================================

from sklearn.cross_validation import train_test_split
# split the data with 50% in each set
X1, X2, y1, y2 = train_test_split(X, y, random_state=0,
                                  train_size=0.5)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)

# fit the model on one set of data
model.fit(X1, y1)

# evaluate the model on the second set of data
y2_model = model.predict(X2)

from sklearn.metrics import accuracy_score
accuracy_score(y2, y2_model)

# =============================================================================
# Model validation via cross-validation
# =============================================================================

#Here we do two validation trials, alternately using each half of the data as a holdout set
y2_model = model.fit(X1, y1).predict(X2)
y1_model = model.fit(X2, y2).predict(X1)
accuracy_score(y1, y1_model), accuracy_score(y2, y2_model)

#Here we split the data into five groups
#and we can use Scikit-Learn's cross_val_score convenience routine to do it
from sklearn.cross_validation import cross_val_score
cross_val_score(model, X, y, cv=5)

#Leave-one-out cross validation, and can be used as follows #extreme
from sklearn.cross_validation import LeaveOneOut
scores = cross_val_score(model, X, y, cv=LeaveOneOut(len(X)))
scores

# =============================================================================
# Selecting the Best Model
# =============================================================================
#Validation curves in Scikit-Learn

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
#we will use this method later
def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree),
                         LinearRegression(**kwargs))
    

#let's create some data to which we will fit our model
import numpy as np

def make_data(N, err=1.0, rseed=1):
    # randomly sample the data
    rng = np.random.RandomState(rseed)
    X = rng.rand(N, 1) ** 2
    y = 10 - 1. / (X.ravel() + 0.1)
    if err > 0:
        y += err * rng.randn(N)
    return X, y

X, y = make_data(40)

#visualize our data
import matplotlib.pyplot as plt
import seaborn; seaborn.set()  # plot formatting

X_test = np.linspace(-0.1, 1.1, 500)[:, None]

plt.scatter(X.ravel(), y, color='black')
axis = plt.axis()
for degree in [1, 3, 5]:
    y_test = PolynomialRegression(degree).fit(X, y).predict(X_test)
    plt.plot(X_test.ravel(), y_test, label='degree={0}'.format(degree))
plt.xlim(-0.1, 1.0)
plt.ylim(-2, 12)
plt.legend(loc='best');

'''
We can make progress in this by visualizing the validation curve for this 
particular data and model; this can be done straightforwardly using the 
validation_curve 
convenience routine provided by Scikit-Learn. Given a model, data, parameter name, 
and a range to explore, this function will automatically compute both 
the training score and validation score across the range
'''
from sklearn.learning_curve import validation_curve
degree = np.arange(0, 21)
train_score, val_score = validation_curve(PolynomialRegression(), X, y,
                                          'polynomialfeatures__degree', degree, cv=7)

plt.plot(degree, np.median(train_score, 1), color='blue', label='training score')
plt.plot(degree, np.median(val_score, 1), color='red', label='validation score')
plt.legend(loc='best')
plt.ylim(0, 1)
plt.xlabel('degree')
plt.ylabel('score');

# =============================================================================
# Validation in Practice: Grid Search
# =============================================================================

#automated tools to do this in the grid search
from sklearn.grid_search import GridSearchCV

param_grid = {'polynomialfeatures__degree': np.arange(21),
              'linearregression__fit_intercept': [True, False],
              'linearregression__normalize': [True, False]}

grid = GridSearchCV(PolynomialRegression(), param_grid, cv=7)

grid.fit(X, y);

#Now that this is fit, we can ask for the best parameters as follows
grid.best_params_

model = grid.best_estimator_

plt.scatter(X.ravel(), y)
lim = plt.axis()
y_test = model.fit(X, y).predict(X_test)
plt.plot(X_test.ravel(), y_test, hold=True);
plt.axis(lim);



