# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 19:57:49 2018

@author: SK056042
"""

import pandas as pd
from pandas import read_csv 
import os
import math
import numpy as np
import statistics
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

import seaborn as sns
import lightgbm as lgb
import catboost as cb

os.chdir("C:/Users/SK056042/Desktop/Kaggle/AV")

# train and test are loaded after adding manual features.
train=read_csv("train_data.csv")
test=read_csv("test_data.csv")

traindata=pd.get_dummies(train)
testdata=pd.get_dummies(test)

train_data=traindata.drop('renewal',axis=1)
target=traindata['renewal']

oversampler=SMOTE(random_state=0, ratio=0.7)
smf_features,smf_labels=oversampler.fit_sample(train_data,target)

dtrain = lgb.Dataset(smf_features, label=smf_labels)


params = {}
params['learning_rate'] = 0.001
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 10
params['max_depth'] = 10


clf = lgb.train(params, dtrain, 100)

pred=clf.predict(testdata)



