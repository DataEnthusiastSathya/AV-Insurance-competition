# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 01:41:47 2018

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

import pickle
import seaborn as sns
import lightgbm as lgb
import catboost as cb
sns.set()
plt.rcParams["figure.figsize"] = [16,9]

os.chdir("C:/Users/SK056042/Desktop/Kaggle/AV")


train=read_csv("train_data.csv")
test=read_csv("test_data.csv")

train=train.fillna(0)
traindata=pd.get_dummies(train)

testdata=pd.get_dummies(test)
#train.shape
#Out[199]: (79853, 21)

#traindata.shape
#Out[200]: (79853, 26)


train_data=traindata.drop('renewal',axis=1)
target=traindata['renewal']


train_x, test_x, train_y, test_y = train_test_split(traindata, target, test_size = 0.30, random_state = 42)

train_x.shape#(55897, 26)
test_x.shape#(23956, 26)
train_y.shape#(55897,)
test_y.shape#(23956,)



train_x=traindata.drop('renewal', axis=1)   
train_y=traindata['renewal']    

oversampler=SMOTE(random_state=0, ratio=0.7)
sm_features,sm_labels=oversampler.fit_sample(train_x,train_y)
    

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


----------------------------------
clf = lgb.train(params, dtrain, 100)
#clf1 = lgb.train(params1, dtrain, 100)

testdata=pd.get_dummies(test)
pred=clf.predict(testdata)

filename = 'lgbm_model.sav'
pickle.dump(clf, open(filename, 'wb'))
----------------------------

--------------
pred=pd.DataFrame(pred)
pred.to_csv('predlgbm.csv')

pred1=pd.DataFrame(pred1)
pred1.to_csv('predlgbmss.csv')



y_pred=clf1.predict(test_x)


confusion_matrix(test_y,y_pred)



def my_func(a):
    print(a)
    if a >= 0.5:
        b=1
    else:
        b=0

    return b



y_pred=pd.DataFrame(y_pred)

test_y=pd.DataFrame(test_y)
test_y.to_csv("testy.csv")
pred=read_csv("predop.csv")
confusion_matrix(pred.renewal,pred.Pred)

roc_auc_score(pred.renewal,pred.Pred)

-----------------------
testop=clf1.predict(testdata)
testop=pd.DataFrame(testop)
testop.to_csv("testop.csv")


--------------------
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001,
                     solver='adam', verbose=10,  random_state=21,tol=0.000000001)

testdata=testdata.fillna(0)
ssm_features = StandardScaler().fit_transform(sm_features)
testfeat = StandardScaler().fit_transform(testdata)

clf.fit(ssm_features,sm_labels)

#y_pred = clf.predict(x_test)
y_pred  =clf.predict_proba(testfeat)
y_pred=pd.DataFrame(y_pred)
y_pred.to_csv("nnop.csv")
x_pred = clf.predict(sm_features)
