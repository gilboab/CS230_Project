#!/usr/bin/env python
#from bittrex import bittrex
import keras
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, LSTM, TimeDistributed, Convolution1D
from keras.optimizers import SGD, Adam
from sklearn.preprocessing import StandardScaler
import numpy as np
import datetime
import time
import csv
import matplotlib.pyplot as plt
from numpy import genfromtxt
import math
import h5py
#import tensorflow as tf
#from tensorflow.python.framework import ops
#from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

sc = StandardScaler()



def plot_data():
    order_log = np.load('order_log_1.npy')
    plt.plot(order_log[1,:,2],order_log[1,:,1])
    plt.plot(order_log[1,:,5],order_log[1,:,4])
    plt.title('Limit Order Book Snapshot', fontsize=20)
    plt.xlabel('Asset Price')
    plt.ylabel('Accumulative Quantity')
    plt.text(7750, 140, 'bid', fontsize=10)
    plt.text(8400, 40, 'ask', fontsize=10)
    plt.text(8000, -15, 'spread', fontsize=10)
    plt.show()
    return

def test_results (prediction,Y):
    results = np.zeros((4,3))
    for j in range (0,np.shape(Y)[1]):
        for i in range (0,np.shape(Y)[0]):
            if prediction[i,j,0]>prediction[i,j,1] and prediction[i,j,0]>prediction[i,j,2]:
                p = 1
            elif prediction[i,j,2]>prediction[i,j,1] and prediction[i,j,2]>prediction[i,j,0]:
                p = -1
            else:
                p = 0
            if Y[i,j,0]==1:             # increase
                results[0,0] +=1
                if p==1:
                    results[1,0] +=1    # correct
                elif p==0:
                    results[2,0] +=1
                else:
                    results[3,0] +=1
            elif Y[i,j,1]==1:           # no change
                results[0,1] +=1
                if p==0:
                    results[2,1] +=1    # correct
                elif p==1:
                    results[1,1] +=1
                else:
                    results[3,1] +=1
            elif Y[i,j,2]==1:           # decrease
                results[0,2] +=1
                if p==-1:
                    results[3,2] +=1    # correct
                elif p==1:
                    results[1,2] +=1
                else:
                    results[2,2] +=1
                    

    print (results)
    accuracy = (results[1,0]+results[2,1]+results[3,2])/(results[0,0]+results[0,1]+results[0,2])
    print (accuracy)
    return ()

def simulate_results (prediction,Y, B):
    print (np.shape(prediction))
    print (np.shape(Y))
    print (np.shape(B))
    usd=0
    btc=0
    init_flag = 1
    value = np.zeros(np.shape(B))
    p=0
    for i in range (0,np.shape(Y)[0]):
        if prediction[i,0]>prediction[i,1] and prediction[i,0]>prediction[i,2]:
            p = 1
        elif prediction[i,2]>prediction[i,1] and prediction[i,2]>prediction[i,0]:
            p = -1
        else:
            p = 0
        if p==1:
            if init_flag==1:
                btc = 1
                init_flag = 0
            else:
                if btc==0:
                    btc = usd/B[i,0]
                    usd = 0
        elif p==-1:
            if init_flag==1:
                usd = B[i,0]
                init_flag=0
            else:
                if usd==0:
                    usd = btc*B[i,0]
                    btc = 0
        value[i,0] = usd + btc*B[i,0]
        if value[i,0] == 0:
            value[i,0] = B[0,0] # No transaction yet. Dont want to show zero
            
    print (B[0,0], value[-1,0])
    plt.plot(B[:,0],label='Bitcoin')
    plt.plot(value[:,0],label='trading performance')
    plt.legend()
    plt.title('Prediction Simulation', fontsize=20)
    plt.xlabel('samples')
    plt.ylabel('Value')
    plt.show()

if __name__ == '__main__':
#    order_100()
#    exit()
#    plot_data()
#    exit()
    drop = 0.2
    X = np.load('../dataset/X_new_100.npy')
    Y_tmp = np.load('../dataset/Y_oh_new_100.npy')
    B = np.load('../dataset/B_new_100.npy')
    m=25000
    X = X[:,0:m]
    B = B[:,0:m]
    Y_tmp = Y_tmp[:,:,:,0:m]
    Y = np.zeros((3,np.shape(Y_tmp)[3]))
    Y[:,:] = Y_tmp[1,1,:,:] # Y [time step (1,2,3,5,10) , threshold (0.1%, 0.2%, 0.3%, 0.4%, 0.5%) , one hot (incr, no change, decrease), m
#   Majority voting results
#    for i in range (0,np.shape(Y_tmp)[-1]):
#        if np.sum(Y_tmp[:,1,0,i])>2 and np.sum(Y_tmp[:,1,2,i])<2:
#            Y[:,i] = [1,0,0] # increase
#        elif  np.sum(Y_tmp[:,1,0,i])<2 and np.sum(Y_tmp[:,1,2,i])>2:
#            Y[:,i] = [0,0,1] # decrease
#        else:
#            Y[:,i] = [0,1,0] # no change

    m = np.shape(X)[1]
    # shuffle
    s = np.arange(m)
    np.random.shuffle(s)
#    X=X[:,s]
#    Y=Y[:,s]
    X = X.T
    Y = Y.T
    B = B.T
    X = sc.fit_transform(X) # normalization
    window = 4
    m = int(np.floor(m/window)*window)
    X = X[0:m,:]
    Y = Y[0:m,:]
    B = B[0:m,:]
#    train_size = int(np.floor(m*0.9/10)*10)
    train_size = int(np.floor(m*0.9/10)*10)
    dev_size = m-train_size
    X_train = np.zeros((train_size-window,window,200))
    X_dev = np.zeros((dev_size-window,window,200))
    Y_train = np.zeros((train_size-window,3))
    Y_dev = np.zeros((dev_size-window,3))
    B_train = np.zeros((train_size-window,1))
    B_dev = np.zeros((dev_size-window,1))

    for i in range (0,train_size-window):
        for j in range (0,window):
            X_train[i,j,:] = X[i+j,:]
        Y_train[i,:] = Y[i+window,:]    
        B_train[i,:] = B[i+window,0]
    for i in range (0,dev_size-window):
        for j in range (0,window):
            X_dev[i,j,:] = X[i+train_size+j,:]
        Y_dev[i,:] = Y[i+train_size+window,:]
        B_dev[i,:] = B[i+train_size+window,0]
 
    print (np.shape(X_train))
    print (np.shape(Y_train))
    print (np.shape(B_train))
#    model = load_model('lstm_model_th_candidate.h5')

    model = load_model('lstm_model_th_bidirectional_candidate.h5')
#    model = load_model('lstm_model_th_bidirectional.h5')
#    prediction = model.predict(X_train, batch_size=None, verbose=0, steps=None)
#    test_results(np.reshape(prediction,(np.shape(prediction)[0],1,3)),np.reshape(Y_train,(np.shape(Y_train)[0],1,3)))
    prediction = model.predict(X_dev, batch_size=None, verbose=0, steps=None)
    test_results(np.reshape(prediction,(np.shape(prediction)[0],1,3)),np.reshape(Y_dev,(np.shape(Y_dev)[0],1,3)))
    simulate_results(prediction, Y_dev,B_dev)
