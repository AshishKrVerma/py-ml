# -*- coding: utf-8 -*-
"""
Created on Sat May 19 18:11:04 2018

@author: averma


Problem Statement:

Build a machine learning model which can predict the survival (independent/ target variable) 
of the passengers from the characteristics such as gender, class, etc. from the dataset.

Description of the Problem:

• The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. 
On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, 
killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the 
international community and led to better safety regulations for ships.

• One of the reasons that the shipwreck led to such loss of life was that there
 were not enough lifeboats for the passengers and crew. Although there was some
 element of luck involved in surviving the sinking, some groups of people were 
 more likely to survive than others, such as women, children, and the upper-class.

Dataset:

Dataset has been split into two groups.

• The Training set: It’ll be used to build your machine learning models. 
For the training set, we provide the target variable showing the survival of each passenger. 
Your model will be based on “features” like passengers’ gender and class. 
This dataset is what you use to craft your dataset.

• The Scoring set: It’ll be used to score your model. For this dataset we provide 
the “features”, but not the target variable, survival. Use this to score your dataset 
and generate predictions as to whether the passenger survived or not.

• Data Dictionary also has been provided.

• In this challenge, predict which people were more likely to survive based on 
their characteristics. In particular, we ask you to apply the tools of machine 
learning to create a model which predicts which passengers survived the tragedy. 
Download the Dataset from below mentioned path:


"""
import numpy as np
import pandas as pd


dataFilePatch="C:/Users/averma/Downloads/AWS/ML/Files"
train = pd.read_csv(dataFilePatch+'/TRAINING.csv')
test = pd.read_csv(dataFilePatch+'/SCORING.csv')

train.isna().any()
test.isna().any()

#fix nan in train
train['Sex']=train['Sex'].apply(lambda sex: 0 if sex == "male" else 1)
train['Age'].fillna((train['Age'].mean()), inplace=True)
train['Cabin'].fillna('0', inplace=True)
train['Cabin']=train['Cabin'].apply(lambda cabin: 0 if cabin == "0" else 1)

##fix nan in test
#test['Age'].fillna((test['Age'].mean()), inplace=True)
#test['Cabin'].fillna('0', inplace=True)
columns = ["Fare", "Pclass","Sex","Age","SibSp","Parch","Cabin"]
Y=train['Survived'].values
X=train[list(columns)].values

from sklearn.linear_model import LogisticRegression
model = LogisticRegression();
model.fit(X,Y)

#fix test
test['Sex']=test['Sex'].apply(lambda sex: 0 if sex == "male" else 1)
test['Age'].fillna((test['Age'].mean()), inplace=True)
test['Cabin'].fillna('0', inplace=True)
test['Cabin']=test['Cabin'].apply(lambda cabin: 0 if cabin == "0" else 1)

X_test=test[list(columns)].values

pred=model.predict(X_test)

test = pd.read_csv(dataFilePatch+'/SCORING.csv')
test['Predicted']=pred

test.to_csv(dataFilePatch+"/results.csv",index=False)


