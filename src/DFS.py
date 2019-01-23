# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 12:54:37 2018

@author: Erik

Main deep feature selection class.  This constructs a DFS model that is essntially just a special type of Keras Sequential
model.


"""

from keras import Sequential
from keras.layers import Dense
from OneToOne import OneToOne
from keras.optimizers import SGD, Adam
from keras.regularizers import l1, l2
from ElasticNetRegularizer import ElasticNetRegularizer
import numpy as np
import matplotlib.pyplot as plt


class DFS(Sequential):
    '''
    num_classes - integer. if there are 6 classes, you would just enter 6
    hidden_layers - iterable.  If you want a 128 node hidden layer followed by a 64 node hidden layer, you would put [128, 64]
    in_shape - tuple.  Shape of the input.  If you have 100 features you would just put (100, )
    
    These are the regularization terms, set to the defaults from the study if nothing is specified.
    lambda1 - double. Penalty for selecting a lot of features
    lambda2 - double between 0 and 1.  Trade off between l2 regularization and l1 regularization for input layer only. 
        Defined as (1 - lambda2) l2_norm + lambda2 * l1_norm
    alpha1 - double. same as lambda1 but for the weights of the model excluding the input layer
    alpha2 = double between 0 and 1.  Same as lambda 2 but for trade off on the model side.
    learning_rate - double. used in stochastic gradient decent.  Controls step size.
    
    '''
    def __init__(self, 
                 in_dim, 
                 num_classes,
                 hidden_layers = [128, 64], 
                 lambda1 = 0.003,
                 lambda2 = 1,
                 alpha1 = 0.0001,
                 alpha2 = 0,
                 learning_rate = 0.1,
                 hidden_layer_activation = 'sigmoid',
                 output_layer_activation = 'softmax',
                 loss_function = 'categorical_crossentropy',
                 addl_metrics = ['accuracy']):
        
        
        
        super().__init__()
        
        #store for copying
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.learning_rate = learning_rate
        self.hidden_layer_activation = hidden_layer_activation
        self.output_layer_activation = output_layer_activation
        self.loss_function = loss_function
        self.addl_metrics = addl_metrics
        self.add(
                OneToOne(in_dim, 
                         name = 'input', 
                         input_dim = in_dim, 
                         use_bias = False,
                         kernel_regularizer = ElasticNetRegularizer(lambda1, lambda2)
                         )
                )
        
        for i, num_nodes in enumerate(hidden_layers):
            self.add(
                    Dense(num_nodes, 
                          name = 'layer' + str(i), 
                          activation = hidden_layer_activation, 
                          kernel_regularizer = ElasticNetRegularizer(alpha1, alpha2)
                          )
                    
                    )
        self.add(Dense(num_classes, name = 'output', 
                       activation = output_layer_activation,
                       kernel_regularizer = ElasticNetRegularizer(alpha1, alpha2)
                       )
                    )
        
        
        self.compile(optimizer = SGD(lr = learning_rate, momentum = 0.1),
              loss = loss_function,
              metrics = addl_metrics)

    def get_input_weights(self):
        wts = self.get_layer('input').get_weights()[0]
        return wts.reshape(len(wts)) #convert from column vector to row vector
    
    '''
    handles categorical and binary output.  So, y must be one hot encoded or already in 0/1 format
    '''
    def accuracy(self, X, y):
        pred = self.predict(X)
        
        #categorical case
        if len(y[0] > 1):
            #translate prediction
            pred = np.argmax(pred, axis = 1)
            y = np.argmax(y, axis = 1)
        
        #binary case
        else:
            pred = np.round(pred)
        
        return np.sum(pred == y) / len(y)
    
    def show_bar_chart(self):
        wts = self.get_input_weights() #get raw data from neural net
        wts = np.abs(wts)
        y_pos = np.arange(len(wts))
        plt.bar(y_pos, wts)
        plt.show()
        
    def get_weight_feature_tuples(self, features):
        weights = self.get_input_weights()
        return list(zip(features, weights))
                
    
    def get_top_features(self, num_features, features):
        def get_weight(e):
            return e[1]
        weights_features = self.get_weight_feature_tuples(features)
        sorted_weights = sorted(weights_features, key = get_weight, reverse = True)
        return sorted_weights[0:num_features]
    
    def write_features(self, file_name, lambda1):
        weights = self.get_input_weights()
        file = open(file_name, 'a')
        file.write(str(lambda1) + ',')
        for weight in weights:
            file.write(str(weight) + ",")
        file.write('\n')
        file.close()
    
    def write_predictions(self, file_name, X, lambda1):
        file = open(file_name, 'a')
        y_pred = self.predict(X)
        #check if it is a regression model or a classificaion model
    
        if len(y_pred[0]) == 1: #regression case
            y_pred.reshape(len(y_pred))
        else: # classification case
            y_pred = np.argmax(y_pred, axis = 1)
            
        file.write(str(lambda1) + ',')
        for i in range(len(y_pred)):
            file.write(str(y_pred[i]) + ",")
        file.write('\n')
        file.close()
    
    def write_true_pred(file_name, y_true, y_pred):
        file = open(file_name, 'w')
        file.write("True,Predicted\n")
        for i in range(len(y_true)):
            file.write(str(y_true[i]) + "," + str(y_pred[i]) + "\n")
            
        file.close()
        
    def write_weights(self, file_name, columns):
        weights = self.get_input_weights()
        file = open(file_name, 'w')
        file.write("feature,weight\n")
        for i in range(len(weights)):
            file.write(columns[i] + "," + str(weights[i]) + "\n")
        file.close()
    
    
    def deepcopy(shell_model):
        return DFS(shell_model.in_dim,
                        shell_model.num_classes,
                        hidden_layers = shell_model.hidden_layers, 
                         lambda1 = shell_model.lambda1,
                         lambda2 = shell_model.lambda2,
                         alpha1 = shell_model.alpha1,
                         alpha2 = shell_model.alpha2,
                         learning_rate = shell_model.learning_rate,
                         hidden_layer_activation = shell_model.hidden_layer_activation,
                         output_layer_activation = shell_model.output_layer_activation,
                         loss_function = shell_model.loss_function,
                         addl_metrics = shell_model.addl_metrics
                        )
    
    def output_multi_lambda(shell_model, feature_file_name, prediction_file_name, lambda1s, X_test, y_test, X_train, y_train, X_val, y_val, epochs, batch_size):
        #set up feature file.  Just need lambda 1 + all features
        file = open(feature_file_name, 'w')
        file.write('lambda1,')
        for feat in X_test.columns:
            file.write(feat + ",")
        file.write('\n')
        file.close()
        
        #set up prediction file, just need lambda1 then all test examples
        file = open(prediction_file_name, 'w')
        file.write('lambda1,')
        for i in range(len(y_test)):
            file.write('example_' + str(i) + ',')
        file.write('\n')
        
        #now print y_true
        file.write('True_label,')
        for y in np.argmax(y_test, axis = 1):
            file.write(str(y) + ',')
        file.write('\n')
        file.close()
            
        for lda in lambda1s:
            #take straight copy
            model = DFS.deepcopy(shell_model)
            #overwrite lambda1 value
            model.get_layer('input').kernel_regularizer.lambda1 = lda #set lambda value equal to the current one
            model.fit(x = X_train, y = y_train, epochs = epochs, batch_size = batch_size, validation_data = [X_val, y_val])
            model.write_features(feature_file_name, lda)
            model.write_predictions(prediction_file_name, X_test, lda)
        
    
                
                
            
            