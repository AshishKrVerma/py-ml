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


