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
import warnings
warnings.filterwarnings("ignore")

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
train=train[train['Count_3-6_months_late'].notnull()]#74130
train=train[train['Income']<1000000]
#raw=train

traindata1=pd.get_dummies(train)
traindata=traindata1[['Count_3-6_months_late','Count_6-12_months_late','Count_more_than_12_months_late','sourcing_channel_A','perc_premium_paid_by_cash_credit','residence_area_type_Rural','residence_area_type_Urban','Amt_paid_per_Premium','id','renewal']]


testdata1=pd.get_dummies(test)
testdata=testdata1[['Count_3-6_months_late','Count_6-12_months_late','Count_more_than_12_months_late','sourcing_channel_A','perc_premium_paid_by_cash_credit','residence_area_type_Rural','residence_area_type_Urban','Amt_paid_per_Premium']]



params = {}
params['learning_rate'] = 0.001
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.7
params['num_leaves'] = 5
params['min_data'] = 10
params['max_depth'] = 15
params['feature_fraction'] = 0.8
params['min_data_in_leaf']=30





#traindata=pd.get_dummies(train)
#x_train=traindata.drop('renewal',axis=1)
#y_train=traindata['renewal']

oversampler=SMOTE(random_state=0, ratio=1)

rocm=[1,2,3,4,5,6,7,8,9,10]
num_folds = 10
subset_size = math.floor(len(traindata)/num_folds)
for i in range(num_folds):
    print(i)
    training = traindata[:i*subset_size]. append(traindata[(i+1)*subset_size:])
    train_data=training.drop(['id','renewal'], axis=1)
    sm_x,sm_y=oversampler.fit_sample(train_data,training['renewal'])
    
    #print(sm_y.sum()/len(sm_y))
    lgtrain=lgb.Dataset(sm_x,label=sm_y)
    clf = lgb.train(params, lgtrain, 700)

    
    testing = traindata[i*subset_size:][:subset_size]
    testd=testing.drop(['renewal','id'], axis=1)
    testd=pd.get_dummies(testd)
    
    lgbpred=clf.predict(testd)
    for x in range(0,len(lgbpred)):
        if lgbpred[x]>=.5:       # setting threshold to .5
           lgbpred[x]=1
        else:  
           lgbpred[x]=0
    roc=roc_auc_score(testing['renewal'],lgbpred)
    rocm[i]=roc
    print(roc)
print(sum(rocm)/10)

#0.7025592002600546
#0.7155843125153101 traindata=traindata1[['Count_3-6_months_late','Count_6-12_months_late','Count_more_than_12_months_late','sourcing_channel_A','perc_premium_paid_by_cash_credit','residence_area_type_Rural','residence_area_type_Urban','Amt_paid_per_Premium','sourcing_channel_B','Premium_Paid','Premium_Pending','sourcing_channel_C','id','renewal']]
#0.7170857359336071 traindata=traindata1[['Count_3-6_months_late','Count_6-12_months_late','Count_more_than_12_months_late','sourcing_channel_A','perc_premium_paid_by_cash_credit','residence_area_type_Rural','residence_area_type_Urban','Amt_paid_per_Premium','sourcing_channel_B','Premium_Paid','Premium_Pending','id','renewal']]
#0.7216170263936575 traindata=traindata1[['Count_3-6_months_late','Count_6-12_months_late','Count_more_than_12_months_late','sourcing_channel_A','perc_premium_paid_by_cash_credit','residence_area_type_Rural','residence_area_type_Urban','Amt_paid_per_Premium','sourcing_channel_B','Premium_Paid','id','renewal']]
#0.7294251670384553 traindata=traindata1[['Count_3-6_months_late','Count_6-12_months_late','Count_more_than_12_months_late','sourcing_channel_A','perc_premium_paid_by_cash_credit','residence_area_type_Rural','residence_area_type_Urban','Amt_paid_per_Premium','sourcing_channel_B','id','renewal']]
#0.7296266202718599 traindata=traindata1[['Count_3-6_months_late','Count_6-12_months_late','Count_more_than_12_months_late','sourcing_channel_A','perc_premium_paid_by_cash_credit','residence_area_type_Rural','residence_area_type_Urban','Amt_paid_per_Premium','id','renewal']]



lg=clf.predict(testdata)

for x in range(0,len(lg)):
    if lg[x]>=.5:       # setting threshold to .5
       lg[x]=1
    else:  
       lg[x]=0

lg=pd.DataFrame(lg)
lg.to_csv("lg.csv")        
-----------------------------------

    



lgtrain=lgb.Dataset(traindata.drop(['id','renewal'],axis=1),traindata['renewal'])
clf = lgb.train(params, lgtrain, 20000)

ypred=clf.predict(testdata)
ypred=pd.DataFrame(ypred)
ypred.to_csv("yprednew.csv")
