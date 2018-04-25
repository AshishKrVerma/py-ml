# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 14:14:16 2018

@author: averma
"""

# =============================================================================
# Naive Bayes Classification
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# =============================================================================
# Gaussian Naive Bayes
# =============================================================================

#the assumption is that data from each label is drawn from a simple Gaussian distribution

#make data
from sklearn.datasets import make_blobs
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu');

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, y);

#Now let's generate some new data and predict the label:
rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
ynew = model.predict(Xnew)

#we can plot this new data to get an idea of where the decision boundary is.

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
lim = plt.axis()
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
plt.axis(lim);

# =============================================================================
# Example: Classifying Text
# =============================================================================

from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups()
data.target_names

#we will select just a few of these categories, and download the training and testing set
categories = ['talk.religion.misc', 'soc.religion.christian',
              'sci.space', 'comp.graphics']
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

print(train.data[5])

# convert the content of each string into a vector of numbers.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

model = make_pipeline(TfidfVectorizer(), MultinomialNB())

#With this pipeline, we can apply the model to the training data, and predict labels for the test data
model.fit(train.data, train.target)
labels = model.predict(test.data)

#confusion matrix between the true and predicted labels for the test data:
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label');
#Evidently, even this very simple classifier can successfully separate space talk from computer talk, 
#but it gets confused between talk about religion and talk about Christianity. 

# Here's a quick utility function that will return the prediction for a single string:
def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]


predict_category('sending a payload to the ISS')
predict_category('discussing islam vs atheism')
predict_category('determining the screen resolution')


