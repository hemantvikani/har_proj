# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 13:45:55 2018

@author: heman
"""

# Importing the libraries
import pandas as pd

print('hemant')
print('komal')

# Importing the dataset
df1 = pd.read_csv("train.csv")

df2= pd.read_csv("test.csv")


features_train=df1.iloc[:,0:-1].values
labels_train=df1.iloc[:,-1:].values
features_test=df2.iloc[:,0:-1].values
labels_test=df2.iloc[:,-1:].values



#checking NAN values
df1.isnull().values.any()
df2.isnull().values.any()


#all algorithms without feature selection


#decision tree entropy
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion = "entropy" ,random_state=0)
classifier.fit(features_train,labels_train)

labels_pred_entropy_dec= classifier.predict(features_test)
print labels_pred_entropy_dec
score_entropy_dec=classifier.score(features_test,labels_test)
print score_entropy_dec


#decision tree gini
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion = "gini" ,random_state=0)
classifier.fit(features_train,labels_train)

labels_pred_gini_dec= classifier.predict(features_test)
print labels_pred_gini_dec
score_gini_dec=classifier.score(features_test,labels_test)
print score_gini_dec



#random forest entropygit
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion = "entropy",random_state=0)
classifier.fit(features_train,labels_train)

labels_pred_entropy_random= classifier.predict(features_test)
print labels_pred_entropy_random
score_entropy_random=classifier.score(features_test,labels_test)
print score_entropy_random


#random forest gini
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion = "gini",random_state=0)
classifier.fit(features_train,labels_train)

labels_pred_gini_random= classifier.predict(features_test)
print labels_pred_gini_random
score_gini_random=classifier.score(features_test,labels_test)
print score_gini_random



#logistic regression
from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression(random_state=0)
classifier.fit(features_train,labels_train)

labels_pred_log= classifier.predict(features_test)
print labels_pred_log
score_log=classifier.score(features_test,labels_test)
print score_log


#fitting K-NN to the training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,p=2)
classifier.fit(features_train,labels_train)



labels_pred_knn=classifier.predict(features_test)
print labels_pred_knn
score_knn=classifier.score(features_test,labels_test)
print score_knn


#max_score in without feature selection algorithms
somelist =  [score_entropy_dec,score_gini_dec,score_entropy_random,score_gini_random,score_log,score_knn] 
max_value = max(somelist)
ind = somelist.index(max_value)

print "score using logistic regression algorithm is best and the  score is ",max_value


#all algorithms with feature selection

df_row_merged = pd.concat([df1, df2], ignore_index=True)
features=df_row_merged.iloc[:,0:-1].values
labels=df_row_merged.iloc[:,-1].values


#np.asarray(labels).dtype


#feature selection
from sklearn.decomposition import PCA
pca=PCA(n_components=65)                              #first 65 features are selected to get best result
features=pca.fit_transform(features)

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 0)

explained_variance=pca.explained_variance_ratio_


#decision tree entropy
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion = "entropy" ,random_state=0)
classifier.fit(features_train,labels_train)

labels_pred_entropy_dec_fs= classifier.predict(features_test)
print labels_pred_entropy_dec_fs
score_entropy_dec_fs=classifier.score(features_test,labels_test)
print score_entropy_dec_fs


#decision tree gini
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion = "gini" ,random_state=0)
classifier.fit(features_train,labels_train)

labels_pred_gini_dec_fs= classifier.predict(features_test)
print labels_pred_gini_dec_fs
score_gini_dec_fs=classifier.score(features_test,labels_test)
print score_gini_dec_fs



#random forest entropy
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion = "entropy",random_state=0)
classifier.fit(features_train,labels_train)

labels_pred_entropy_random_fs= classifier.predict(features_test)
print labels_pred_entropy_random_fs
score_entropy_random_fs=classifier.score(features_test,labels_test)
print score_entropy_random_fs

#random forest gini
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion = "gini",random_state=0)
classifier.fit(features_train,labels_train)

labels_pred_gini_random_fs= classifier.predict(features_test)
print labels_pred_gini_random_fs
score_gini_random_fs=classifier.score(features_test,labels_test)
print score_gini_random_fs



#logistic regression
from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression(random_state=0)
classifier.fit(features_train,labels_train)

labels_pred_log_fs= classifier.predict(features_test)
print labels_pred_log_fs 
score_log_fs=classifier.score(features_test,labels_test)
print score_log_fs



#fitting K-NN to the training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,p=2)
classifier.fit(features_train,labels_train)



labels_pred_knn_fs=classifier.predict(features_test)
print labels_pred_knn_fs
score_knn_fs=classifier.score(features_test,labels_test)
print score_knn_fs

somelist1 =  [score_entropy_dec_fs,score_gini_dec_fs,score_entropy_random_fs,score_gini_random_fs,score_log_fs,score_knn_fs] 
max_value1 = max(somelist1)
ind1 = somelist1.index(max_value1)
print "score using knn algorithm with feature selection is best and the score is ",max_value1

list2=[score_knn_fs,score_log]
max_value2=max(list2)
ind2=list2.index(max_value2)
print "algorithms using feature selection is better than without feature selection"


#plotting graph
# 1.graph without feature selection

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 

values=[85.06,90.97,86.39,90.70,80.96,96.16]

labels=["score_entropy_dec","score_entropy_random","score_gini_dec","score_gini_random","score_knn","score_log"]
y_pos = np.arange(len(labels))

plt.barh(y_pos, values, align='center', alpha=0.5)
plt.yticks(y_pos, labels)
plt.xlabel('score')
plt.title('graph of scores of algorithms without feature  selection')
 
plt.show()


# 2.graph with feature selection

values=[86.24,90.97,84.53,91.22,96.37,96.86]

labels=["score_entropy_dec_fs","score_entropy_random_fs","score_gini_dec_fs","score_gini_random_fs","score_knn_fs","score_log_fs"]
y_pos = np.arange(len(labels))

plt.barh(y_pos, values, align='center', alpha=0.5)
plt.yticks(y_pos, labels)
plt.xlabel('score')
plt.title('graph of scores of algorithms with feature  selection')
 
plt.show()





