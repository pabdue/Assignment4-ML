#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #4
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

highest_perceptron_accuracy = 0.0
highest_mlp_accuracy = 0.0



for learning_rate in n:

    for shuffle in r:

        #Perceptron
        clf = Perceptron(eta0=learning_rate, shuffle=shuffle, max_iter=1000)
        clf.fit(X_training, y_training)

        accuracy = 0.0
        for x_testSample, y_testSample in zip(X_test, y_test):
            prediction = clf.predict([x_testSample])
            if prediction == y_testSample:
                accuracy += 1.0
        accuracy /= len(X_test)

        if accuracy > highest_perceptron_accuracy:
            highest_perceptron_accuracy = accuracy
            print("Highest Perceptron accuracy so far:", highest_perceptron_accuracy, ", Parameters: learning rate=", learning_rate, ", shuffle=", shuffle)

        #MLP
        clf = MLPClassifier(activation='logistic', learning_rate_init=learning_rate, hidden_layer_sizes=(64,), shuffle=shuffle, max_iter=1000)
        clf.fit(X_training, y_training)

        accuracy = 0.0
        for x_testSample, y_testSample in zip(X_test, y_test):
            prediction = clf.predict([x_testSample])
            if prediction == y_testSample:
                accuracy += 1.0
        accuracy /= len(X_test)

        if accuracy > highest_mlp_accuracy:
            highest_mlp_accuracy = accuracy
            print("Highest MLP accuracy so far:", highest_mlp_accuracy, ", Parameters: learning rate=", learning_rate, ", shuffle=", shuffle)
