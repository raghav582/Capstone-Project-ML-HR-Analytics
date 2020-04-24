#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:56:00 2020

@author: raghav
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv('Desktop/Kaggle/hr-analytics/HR_comma_sep.csv')
dataset.info()
dataset.head()

#data visualization
sns.scatterplot(dataset.average_montly_hours, dataset.satisfaction_level, hue=dataset.left)
sns.barplot(dataset.left, dataset.satisfaction_level)
f, ax = plt.subplots(figsize=(10,5))
sns.countplot(dataset.Department, hue=dataset.left)
sns.countplot(dataset.promotion_last_5years, hue=dataset.left)
sns.countplot(dataset.salary, hue=dataset.left)

#encoding
cat_col=[col for col in dataset.columns if dataset[col].dtype=='object']
dataset=pd.get_dummies(dataset)

#train-test split
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY=train_test_split(dataset.drop(columns=['left']), dataset.left,
                                           test_size=0.1, shuffle=True)

#model
from sklearn.svm import SVC
model=SVC()
sel_col=['satisfaction_level','last_evaluation','average_montly_hours']
model.fit(trainX[sel_col], trainY)
print(model.score(testX[sel_col],testY))    

from sklearn.neural_network import MLPClassifier
model=MLPClassifier()
model.fit(trainX,trainY)

y_predict=model.predict(testX)

print(model.score(testX, testY))