import pandas as pd
import numpy as np
import csv
import json
from datetime import datetime
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier

## Loading the train and test data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.head(10)

print(train.shape)
train.price_range.value_counts()

print(test.shape)

labels = train.price_range
test_ind = test.T_id

# dropping irrelevant columns and separating the labels
train.drop(['price_range'], axis=1, inplace=True)
test.drop(['T_id'], axis=1, inplace=True)


#dealing with null values
train.fillna(0, inplace=True)

#Applying the gradient boosting classifier

clf3 = GradientBoostingClassifier(n_estimators=500, learning_rate=1.0,max_depth=1, random_state=0).fit(train, labels)
x = clf3.predict(test)
scores = cross_val_score(clf3, train, labels, cv=5)
scores.mean() 
df = pd.DataFrame(x)
df.to_csv('sub.csv')