from sklearn.base import clone
from itertools import combinations
import numpy as np 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

## naming convention: _ for variables that are not created upon initialization
class SBS:
    def __init__(self,keep_feat, estimator, scoring = accuracy_score,test_size =0.2, random_state = 1):
        self.keep_feat = keep_feat
        self.estimator = clone(estimator)
        self.scoring= scoring 
        self.test_size = test_size
        self.random_state = random_state
    def input(self,X,y):
        '''
        In this instance, we do the following steps:
        - Have a index of all of the features
        - Have a while loop that removes feature by feature
            - Get a combination of the features index
            - Calculate score within each index combination
            - Grab the indexes of the best scores -> Feed it back to the begining
        '''
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size= self.test_size,random_state= self.random_state)
        feat_numb= X.shape[1]
        self.indices_=tuple(range(feat_numb))
        score_all_dim = self.calc_score(X_train,X_test,y_train,y_test, self.indices_)
        self.score_ = [score_all_dim]
        while feat_numb > self.keep_feat:
            ls_score=[]
            index_subset =[]
            for index in combinations(self.indices_,r = feat_numb -1): # create all combinations of self.indices_. Each combination with length r
                indiv_score = self.calc_score(X_train,X_test,y_train,y_test,index)
                ls_score.append(indiv_score)
                index_subset.append(index)
            best = np.argmax(ls_score) # get the index of the best score
            self.score_.append(ls_score[best])
            feat_numb -= 1
        return self    

    def calc_score(self,X_train,X_test,y_train,y_test,index):
        self.estimator.fit(X_train[:,index],y_train)
        pred = self.estimator.predict(X_test[:,index])
        score = self.scoring(pred,y_test)
        return score


''' 
Lets test it out. Though we have a train_test_split inside of the SBS fuction, we still input our train data set. We can 
call it the **validation dataset**
'''

df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header=None)

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test =train_test_split(X, y, 
                     test_size=0.3, 
                     random_state=0, 
                     stratify=y)


import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

# selecting features
sbs = SBS(estimator=knn, keep_feat=1)
sbs.input(X_train, y_train)

print(sbs.score_)