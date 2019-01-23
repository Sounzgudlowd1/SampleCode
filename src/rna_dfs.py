# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 10:11:52 2018

@author: Erik

This is the main rna analysis.  We used Keras with a tensor flow backend.  I also recommend installing the full Anaconda package
to ensure that all of the code runs properly.

"""

from DFS import DFS
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import resample

def zero_one_norm(X):
    X = X.fillna(0)
    X = (X - X.min()) / (X.max() - X.min()) 
    return X.fillna(0)

#get data and labels
X = pd.read_csv('../data/rna/data.csv')
y_str = pd.read_csv('../data/rna/labels.csv')


#Drop the "Unnamed: 0" column
X = X.drop('Unnamed: 0', axis = 1)
y_str = y_str.drop('Unnamed: 0', axis = 1)

#add y to dataframe for sampling ease
X['Class'] = y_str

#just iterate through 0, 1, 2... for each label
target = 4
all_labels = ['BRCA', 'COAD', 'KIRC', 'LUAD', 'PRAD']

#split X
X_train, X_test= train_test_split(X, test_size = 0.2)

#get the subset of the dataframe 
X_train_minority = X_train[X_train['Class'] == all_labels[target]] #get minority classs, in this case the one we are trying to predict
X_train_majority = X_train[X_train['Class'] != all_labels[target]]

#do balancing on train only, test is test.  
X_train_minority_bal = resample(X_train_minority,
                           replace = True,
                           n_samples = len(X_train_majority)) #upsample until there is a label balance

#after upsampling reconstruct the original data sets
X_train = pd.concat([X_train_minority_bal, X_train_majority])

#after sampling separate out the class again
y_train = X_train['Class']
y_test = X_test['Class']
X_train = X_train.drop('Class', axis = 1)
X_test = X_test.drop('Class', axis = 1)

#zero one normalize
X_train = zero_one_norm(X_train)
X_test = zero_one_norm(X_test)

#set label to "BG" for background if it is NOT the target label
for i, lab in enumerate(all_labels):
    if i != target:
        y_train = y_train.replace(lab, 'BG')
        y_test = y_test.replace(lab, 'BG')
        
print(set(y_train))


#one hot encode
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

#iterate through a variety of regularization values for input layer, first works well, second is over-regularized
lambdas = [0.0001, 0.001]

for lda in lambdas:
    model = DFS(in_dim = len(X_train.columns), 
                num_classes = len(y_train.columns),
                alpha1 = 0.00001, 
                lambda1 = lda,
                learning_rate = 0.05)
    
    model.fit(X_train, y_train, batch_size = 100, epochs = 400)
    
    #for sanity check just print test accuracy
    y_test = np.array(y_test) #easier to report from array rather than dataframe
    model.accuracy(X_test, y_test)
    
    #report the weights for visualizations later
    model.write_weights('../results/rna_weights_' + all_labels[target] + '_' + str(lda) + '.csv', X.columns)
    
    #convert output back to a single label rather than one hot encoded
    y_test = np.argmax(np.array(y_test), axis = 1)
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis = 1)
    
    #just write predictions and true lables for metric reporting
    DFS.write_true_pred('../results/rna_predictions_' + all_labels[target] + '_' + str(lda) + '.csv', y_test, y_pred)
    

    
