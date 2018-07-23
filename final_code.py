# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 23:56:49 2018

@author: SK056042
"""

import pandas as pd
from pandas import read_csv 
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import PolynomialFeatures
import lightgbm as lgb

os.chdir("C:/Users/SK056042/Desktop/Kaggle/AV")

train=read_csv("train_data.csv")
test=read_csv("test_data.csv")

train=train[train['Count_3-6_months_late'].notnull()]#74130

traindata1=pd.get_dummies(train)
traindata=traindata1[['Count_3-6_months_late','Count_6-12_months_late','Count_more_than_12_months_late','sourcing_channel_A','perc_premium_paid_by_cash_credit','residence_area_type_Rural','residence_area_type_Urban','Amt_paid_per_Premium','id','renewal']]

#test[test['Count_3-6_months_late'].isnull()==True]['Count_3-6_months_late']=test[test['Count_3-6_months_late'].notnull()==True]['Count_3-6_months_late'].median()
#test[test['Count_6-12_months_late'].isnull()==True]['Count_6-12_months_late']=test[test['Count_6-12_months_late'].notnull()==True]['Count_6-12_months_late'].median()
#test[test['Count_3-6_months_late'].isnull()==True]['Count_3-6_months_late']=test[test['Count_3-6_months_late'].notnull()==True]['Count_3-6_months_late'].median()
test=test.fillna(0)
test.isnull().sum()

params = {}
params['learning_rate'] = 0.0001
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.7
params['num_leaves'] = 5
params['min_data'] = 10
params['max_depth'] = 15
params['feature_fraction'] = 0.8
params['min_data_in_leaf']=30

oversampler=SMOTE(random_state=0, ratio=1)
-------------------------------------------------------
#approach1- select all features and lgb
train_features, test_features, train_labels, test_labels = train_test_split(traindata1.drop(['id','renewal'],axis=1), traindata.renewal, test_size = 0.25, random_state = 42)

lgtrain=lgb.Dataset(train_features,train_labels)
clf = lgb.train(params, lgtrain, 20000)
ypred=clf.predict(test_features)

for x in range(0,len(ypred)):
    if ypred[x]>0.5:
        ypred[x]=1
    else:
        ypred[x]=0
roc=roc_auc_score(test_labels,ypred)
roc

testdata=pd.get_dummies(test)
testpred=clf.predict(testdata)
testpred=pd.DataFrame(testpred)
testpred.to_csv("output1.csv")

-------------------------------------------------------
#approach-2 selected features and lgb
traindata['Amt_paid_sq']=traindata['Amt_paid_per_Premium']*traindata['Amt_paid_per_Premium']
train_features, test_features, train_labels, test_labels = train_test_split(traindata.drop(['id','renewal'],axis=1), traindata.renewal, test_size = 0.25, random_state = 42)
testd=pd.get_dummies(test)
testdata=testd[['Count_3-6_months_late','Count_6-12_months_late','Count_more_than_12_months_late','sourcing_channel_A','perc_premium_paid_by_cash_credit','residence_area_type_Rural','residence_area_type_Urban','Amt_paid_per_Premium','id']]

sm_x,sm_y=oversampler.fit_sample(train_features,train_labels)

lgtrain=lgb.Dataset(sm_x,sm_y)
    
#lgtrain=lgb.Dataset(train_features,train_labels)
clf = lgb.train(params, lgtrain, 700)
#trainpred=clf.predict(traindata.drop(['id','renewal'],axis=1))
#traindata['lgpred']=trainpred

ypred=clf.predict(test_features)
for x in range(0,len(ypred)):
    if ypred[x]>0.5:
        ypred[x]=1
    else:
        ypred[x]=0
roc=roc_auc_score(test_labels,ypred)
confusion_matrix(test_labels,ypred)

roc



testpred=clf.predict(testdata)
testpred=pd.DataFrame(testpred)
testpred.to_csv("output2.csv")



---------------------------------------------------------
#approach 3 , polynomial followed by PCA and lgb
pf = PolynomialFeatures(degree=2, interaction_only=False,  
                        include_bias=False)
res = pf.fit_transform(traindata.drop(['id','renewal'],axis=1))
pnm=pd.DataFrame(res,columns=pf.get_feature_names(traindata.columns))








