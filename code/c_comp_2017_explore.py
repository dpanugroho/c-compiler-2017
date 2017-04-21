# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 14:21:52 2017

@author: dwipr
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

#%%
np.random.seed(149)

def explore_null_columns(dataFrame):
    for column in dataFrame.columns:
        print(column,dataFrame[column].isnull().sum())

def explore_unique_values(dataFrame):
    for column in dataFrame.columns:
        print(column,len(dataFrame[column].unique()))
        
def waktu_konsultasi_to_epoch(date_str):
    if not len(str(date_str))==1:
        pattern = '%d/%m/%Y'
        epoch = int(time.mktime(time.strptime(date_str, pattern)))
        return epoch
    else: return 0

def waktu_pendaftaran_to_epoch(date_str):
    if not len(str(date_str))==1:
        pattern = '%d/%m/%Y %H.%M'
        epoch = int(time.mktime(time.strptime(date_str, pattern)))
        return epoch
    else: return 0

data_train = pd.read_csv('../data/train.csv')
data_test = pd.read_csv('../data/test.csv')

data_train= data_train.dropna(subset=['Waktu_Konsultasi'], how='any')
data_train_y = data_train['Label']
data_train_x = pd.DataFrame(data_train.drop('Label', axis=1))
merged_train_test = pd.concat([data_train_x,data_test])

for col in merged_train_test.columns:
  merged_train_test[col] = merged_train_test[col].fillna(merged_train_test[col].mode()[0])
merged_train_test.Waktu_Konsultasi = merged_train_test['Waktu_Konsultasi'].apply(waktu_konsultasi_to_epoch)
merged_train_test.Waktu_Pendaftaran = merged_train_test['Waktu_Pendaftaran'].apply(waktu_pendaftaran_to_epoch)
#%%
columns = merged_train_test.columns
cats = [feat for feat in columns if 'Dis' in feat]
categorical_columns = ['Jenis_Kelamin','Hari','Salutation']
categorical_columns += cats

#merged_train_test.drop(categorical_columns, axis=1, inplace=True)

merged_train_test = pd.get_dummies(merged_train_test, columns = categorical_columns)

merged_train_test = (merged_train_test - merged_train_test.mean()) / (merged_train_test.max() - merged_train_test.min())

train_x = merged_train_test[:len(data_train)]
data_test = merged_train_test[len(data_train):]


#%%
train_x, val_x, train_y, val_y = train_test_split(train_x, data_train_y, test_size=0.2)

train_dataset  = train_x
train_dataset['Label'] = train_y

train_dataset= train_dataset.sort_values(by='Label')
train_dataset=train_dataset[:181162]

train_y = train_dataset['Label']
train_x = train_dataset.drop('Label',axis=1)
#%%
logreg = LogisticRegression()
rf = RandomForestClassifier(n_estimators=200)
gnb = GaussianNB()
mlp = MLPClassifier(hidden_layer_sizes=(12, 8,4),verbose=True, learning_rate_init=0.001)

clf = rf
model = clf.fit(train_x, train_y) 
pred_y = model.predict(val_x)

score = accuracy_score(val_y, pred_y)
print(score)





