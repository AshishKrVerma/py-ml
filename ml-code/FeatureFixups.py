# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 13:43:36 2018

@author: averma
"""

data = [
    {'price': 850000, 'rooms': 4, 'neighborhood': 'Queen Anne'},
    {'price': 700000, 'rooms': 3, 'neighborhood': 'Fremont'},
    {'price': 650000, 'rooms': 3, 'neighborhood': 'Wallingford'},
    {'price': 600000, 'rooms': 2, 'neighborhood': 'Fremont'}
]

'''
one-hot encoding, which effectively creates extra columns indicating the presence 
or absence of a category with a value of 1 or 0, respectively. 
When your data comes as a list of dictionaries, Scikit-Learn's 'DictVectorizer' will do this for you.
'''

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False, dtype=int)
vec.fit_transform(data)

vec.get_feature_names() # To see the meaning of each column, you can inspect the feature names

'''
if your category has many possible values, this can greatly increase the size of your dataset.
However, because the encoded data contains mostly zeros, a sparse output can be a very efficient solution:
'''
vec = DictVectorizer(sparse=True, dtype=int)
vec.fit_transform(data)

#TODO - sklearn.preprocessing.OneHotEncoder and sklearn.feature_extraction.FeatureHasher are 
#two additional tools that Scikit-Learn includes to support this type of encoding.

# =============================================================================
# Text Features
# =============================================================================

#1
sample = ['problem of evil',
          'evil queen',
          'horizon problem']

from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer()
X = vec.fit_transform(sample)
X # will not print much

# it is easier to inspect if we convert this to a DataFrame with labeled columns:
import pandas as pd
pd.DataFrame(X.toarray(), columns=vec.get_feature_names())


#2. frequency-inverse document frequency (TF–IDF) which weights the word counts
# by a measure of how often they appear in the documents. The syntax for 
#computing these features is similar to the previous example:

from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
X = vec.fit_transform(sample)
pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

# =============================================================================
# Derived Features
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5])
y = np.array([4, 2, 1, 3, 7])
plt.scatter(x, y);

#still, we can fit a line to the data using LinearRegression and get the optimal result:
from sklearn.linear_model import LinearRegression
X = x[:, np.newaxis]
model = LinearRegression().fit(X, y)
yfit = model.predict(X)
plt.scatter(x, y)
plt.plot(x, yfit);

#but the plot is not quite good though, we need more sophisticated model to describe the relationship between x and y.
#One approach, we can add polynomial features to the data this way:
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3, include_bias=False)
X2 = poly.fit_transform(X)
print(X2)

#The derived feature matrix has one column representing x, and a second column 
#representing x2, and a third column representing x3. 
#Computing a linear regression on this expanded input gives a much closer fit to our data:

model = LinearRegression().fit(X2, y)
yfit = model.predict(X2)
plt.scatter(x, y)
plt.plot(x, yfit);

# =============================================================================
# #Imputation of Missing Data
# =============================================================================

from numpy import nan
X = np.array([[ nan, 0,   3  ],
              [ 3,   7,   9  ],
              [ 3,   5,   2  ],
              [ 4,   nan, 6  ],
              [ 8,   8,   1  ]])
y = np.array([14, 16, -1,  8, -5])

#missing data is nan


from sklearn.preprocessing import Imputer
imp = Imputer(strategy='mean')
X2 = imp.fit_transform(X)
X2

model = LinearRegression().fit(X2, y)
model.predict(X2) # makes no sence. as we are training and predicting the same y :(


# =============================================================================
# Feature Pipelines
# =============================================================================

#Pipeline object, which can be used as follows:

from sklearn.pipeline import make_pipeline

model = make_pipeline(Imputer(strategy='mean'),
                      PolynomialFeatures(degree=2),
                      LinearRegression())

from sklearn.pipeline import make_pipeline

model = make_pipeline(Imputer(strategy='mean'),
                      PolynomialFeatures(degree=2),
                      LinearRegression())

#This pipeline looks and acts like a standard Scikit-Learn object, 
#and will apply all the specified steps to any input data.

model.fit(X, y)  # X with missing values, from above
print(y)
print(model.predict(X))





































