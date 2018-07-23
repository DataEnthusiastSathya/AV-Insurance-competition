# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 20:16:34 2018

@author: SK056042
"""

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
import featuretools as ft

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
train=train[train['Premium_Paid']!=0]
test=read_csv("test_data.csv")



params = {}
params['learning_rate'] = 0.001
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.7
params['num_leaves'] = 5
params['min_data'] = 10
params['max_depth'] = 15


import warnings
warnings.filterwarnings("ignore")
--------------------------------------
traindata=pd.get_dummies(train)
x_train=traindata.drop('renewal',axis=1)
y_train=traindata['renewal']

oversampler=SMOTE(random_state=0, ratio=1)

feature_list = list(train_data.columns)
rocm=[1,2,3,4,5,6,7,8,9,10]
num_folds = 10
subset_size = math.floor(len(traindata)/num_folds)
for i in range(num_folds):
    print(i)
    training = traindata[:i*subset_size]. append(traindata[(i+1)*subset_size:])
    train_data=training.drop('renewal', axis=1)
    sm_x,sm_y=oversampler.fit_sample(train_data,training['renewal'])
    #print(sm_y.sum()/len(sm_y))
    #lgtrain=lgb.Dataset(sm_x,label=sm_y)
    #clf = lgb.train(params, lgtrain, 500)
    rf.fit(sm_x,sm_y)

    importances = list(rf.feature_importances_)
# List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

    testing = traindata[i*subset_size:][:subset_size]
    testd=testing.drop('renewal', axis=1)
    testd=pd.get_dummies(testd)
    
    lgbpred=rf.predict(testd)
    for x in range(0,len(lgbpred)):
        if lgbpred[x]>=.5:       # setting threshold to .5
           lgbpred[x]=1
        else:  
           lgbpred[x]=0
    roc=roc_auc_score(testing['renewal'],lgbpred)
    rocm[i]=roc
    print(roc)
print(sum(rocm)/10)

lg=clf.predict(testdata)

for x in range(0,len(lg)):
    if lg[x]>=.5:       # setting threshold to .5
       lg[x]=1
    else:  
       lg[x]=0

lg=pd.DataFrame(lg)
lg.to_csv("lg.csv")        
    
---------------------------------------------
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 200,verbose=1, random_state = 42)
#rf.fit(train_features, train_labels);

predictions_final = rf.predict_proba(testdata)
predictions_final=pd.DataFrame(predictions_final)







