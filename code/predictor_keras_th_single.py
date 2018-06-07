#!/usr/bin/env python
#from bittrex import bittrex
import keras
from keras import backend as K
from keras.models import Sequential
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



def order_100 ():
    X = np.zeros((200,1))
    Y = np.zeros((5,1))
    epsilon = 0.00000001
    dp = 10 # delta price for binig the data
    for i in range (1,11):
        s = '../dataset/btc_data_log_' + repr (i) + '.npy'
        btc_data = np.load(s)
        m = np.shape(btc_data)[0]-10 # for next sample prediction (1 min to the future)
        s = '../dataset/order_log_' + repr (i) + '.npy'
        order_log = np.load(s)
        mid_price = (order_log[:,0,2]+order_log[:,0,5])/2
        max_ask = (order_log[:,499,2]-mid_price)
        max_bid = (mid_price - order_log[:,499,5])
        print ('max ask',np.max(max_ask))
        print ('max bid',np.max(max_bid))
        for j in range (0,m):
            if j % 100 == 0:
                print (j)
            mid_price = (order_log[j,0,2] + order_log[j,0,5])/2
            x = np.zeros((200,1))
            y = np.zeros((5,1))
            a_ind = 0 # ask index
            b_ind = 0 # bid index
            for k in range (0,100): # run through the orders and bin them to groups of 10
                while a_ind < 500 and order_log[j,a_ind,2] - mid_price < 10*(k+1):
                    x[k,0]+=order_log[j,a_ind,0]
                    a_ind +=1
                while b_ind < 500 and mid_price - order_log[j,b_ind,5] < 10*(k+1):
                    x[100+k,0]+=order_log[j,b_ind,3]
                    b_ind +=1
            while a_ind<500:
                x[99,0] += order_log[j,a_ind,0]
                a_ind +=1
            while b_ind<500:
                x[199,0] += order_log[j,b_ind,3]
                b_ind +=1
            X = np.append(X,x,axis=1)
#            x[0:100,0]   = order_log[j,0:100,0]/(np.abs(order_log[j,0:100,2]-mid_price)+epsilon)
#            x[100:200,0] = order_log[j,0:100,3]/(np.abs(order_log[j,0:100,5]-mid_price)+epsilon)
#            X = np.append(X,x,axis=1)
            if btc_data[j+1,0]>btc_data[j,0]:   # if next price is higher than current price
                y[0,0] = 1
            if btc_data[j+2,0]>btc_data[j,0]:   # if next price is higher than current price
                y[1,0] = 1
            if btc_data[j+3,0]>btc_data[j,0]:   # if next price is higher than current price
                y[2,0] = 1
            if btc_data[j+5,0]>btc_data[j,0]:   # if next price is higher than current price
                y[3,0] = 1
            if btc_data[j+10,0]>btc_data[j,0]:   # if next price is higher than current price
                y[4,0] = 1
            Y = np.append(Y,y,axis=1)
    print (np.shape(Y))
    print (np.shape(X))
    np.save('../dataset/X_100',X)
    np.save('../dataset/Y_100',Y)
#    mid = (order_log[1,0,5]+order_log[1,0,2])/2
#    r = np.arange(mid,mid+1000,10)
#    plt.plot(r,X[0:100,1])
#    r = np.arange(mid, mid-1000,-10)
#    plt.plot(r,X[100:200,1])
#    plt.title('training set features (bins)', fontsize=20)
#    plt.xlabel('Asset Price')
#    plt.ylabel('Quantity')    
#    plt.show()
    return ()

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

#def simulate_results (prediction,Y, B, train_size):




if __name__ == '__main__':
#    order_100()
#    exit()
#    plot_data()
#    exit()
    drop = 0.2
    X = np.load('../dataset/X_new_100.npy')
    Y_tmp = np.load('../dataset/Y_oh_new_100.npy')
    B = np.load('../dataset/B_new_100.npy')
    X = X[:,0:25000]
    Y_tmp = Y_tmp[:,:,:,0:25000]
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
    X = sc.fit_transform(X) # normalization
    window = 4
    m = int(np.floor(m/window)*window)
    X = X[0:m,:]
    Y = Y[0:m,:]
    train_size = int(np.floor(m*0.9/10)*10)
    dev_size = m-train_size
    X_train = np.zeros((train_size-window,window,200))
    X_dev = np.zeros((dev_size-window,window,200))
    Y_train = np.zeros((train_size-window,3))
    Y_dev = np.zeros((dev_size-window,3))
    for i in range (0,train_size-window):
        for j in range (0,window):
            X_train[i,j,:] = X[i+j,:]
        Y_train[i,:] = Y[i+window,:]    
    for i in range (0,dev_size-window):
        for j in range (0,window):
            X_dev[i,j,:] = X[i+train_size+j,:]
        Y_dev[i,:] = Y[i+train_size+window,:]    

 
    print (np.shape(X_train))
    print (np.shape(Y_train))
    model = Sequential()
    '''
    model.add(TimeDistributed(Dense(50, activation='relu'), input_shape=(window, 200)))
    model.add(TimeDistributed(Dropout(drop)))
    model.add(TimeDistributed(Dense(30, activation='relu')))
    model.add(TimeDistributed(Dropout(drop)))
    model.add(TimeDistributed(Dense(20, activation='relu')))
    model.add(TimeDistributed(Dropout(drop)))
#    model.add(TimeDistributed(Dense(10, activation='relu'), dropout = drop))
#    model.add(TimeDistributed(Dense(5, activation='relu'), dropout = drop))

#    model.add(LSTM(256, activation='relu', recurrent_activation='relu', return_sequences=True, dropout=drop))
    model.add(LSTM(256, activation='relu', recurrent_activation='relu', dropout=drop))
#    model.add(TimeDistributed(Dense(3, activation='softmax')))
    model.add(Dense(3, activation='softmax'))
    '''
    model.add(Convolution1D(256, 5, border_mode='same', input_shape=(window, 200)))
#    model.add(TimeDistributed(Dense(128), input_shape=(window, 200)))
    model.add(TimeDistributed(Dropout(drop)))
    model.add(Convolution1D(128, 5, border_mode='same'))
#    model.add(TimeDistributed(Dropout(drop)))
    model.add(Convolution1D(64, 5, border_mode='same'))
#    model.add(TimeDistributed(Dropout(drop)))
    model.add(LSTM(256, return_sequences=True, dropout=drop))
    model.add(LSTM(256, return_sequences=False, dropout=drop))
    model.add(Dense(3, activation='softmax'))
    model.summary()
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
    model.fit(X_train, Y_train,
          epochs=80,
          batch_size=128)
    score = model.evaluate(X_dev, Y_dev, batch_size=128)
#    score = model.evaluate(X_dev, Y_dev, batch_size=128)
    print (score)
    prediction = model.predict(X_train, batch_size=None, verbose=0, steps=None)
#    test_results(prediction,Y_train)
    test_results(np.reshape(prediction,(np.shape(prediction)[0],1,3)),np.reshape(Y_train,(np.shape(Y_train)[0],1,3)))
    prediction = model.predict(X_dev, batch_size=None, verbose=0, steps=None)
#    test_results(prediction,Y_dev)
    test_results(np.reshape(prediction,(np.shape(prediction)[0],1,3)),np.reshape(Y_dev,(np.shape(Y_dev)[0],1,3)))
    model.save('lstm_model_th.h5')
