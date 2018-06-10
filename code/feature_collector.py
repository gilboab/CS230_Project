#!/usr/bin/env python
#from bittrex import bittrex
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
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
    Y_oh = np.zeros((5,5,3,1))
    B = np.zeros((21,1))
    threshold = 0.001
    epsilon = 1
    dp = 10 # delta price for binig the data
    for i in range (1,11):
#    for i in range (1,2):
        s = '../dataset/btc_data_log_' + repr (i) + '.npy'
        btc_data = np.load(s)
        m = np.shape(btc_data)[0]-10 # for next sample prediction (1 min to the future)
        s = '../dataset/order_log_' + repr (i) + '.npy'
        order_log = np.load(s)
        s = '../dataset/history_log_' + repr (i) + '.npy'
        history = np.load(s)
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
            y_oh = np.zeros((5,5,3,1))
            a_ind = 0 # ask index
            b_ind = 0 # bid index
#            for k in range (0,100): # run through the orders and bin them to groups of 10
#                while a_ind < 500 and order_log[j,a_ind,2] - mid_price < 10*(k+1):
#                    x[k,0]+=order_log[j,a_ind,0]
#                    a_ind +=1
#                while b_ind < 500 and mid_price - order_log[j,b_ind,5] < 10*(k+1):
#                    x[100+k,0]+=order_log[j,b_ind,3]
#                    b_ind +=1
            for k in range (0,100): # run through the orders and bin them to groups of 10
                while a_ind < 500 and order_log[j,a_ind,2] - btc_data[j,0] < 0.001*(k+1)*btc_data[j,0]:
                    x[k,0]+=order_log[j,a_ind,0]
                    a_ind +=1
                while b_ind < 500 and btc_data[j,0] - order_log[j,b_ind,5] < 0.001*(k+1)*btc_data[j,0]:
                    x[100+k,0]+=order_log[j,b_ind,3]
                    b_ind +=1
            while a_ind<500:
                x[99,0] += order_log[j,a_ind,0]
                a_ind +=1
            while b_ind<500:
                x[199,0] += order_log[j,b_ind,3]
                b_ind +=1            
            X = np.append(X,x,axis=1)
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
            for th in range (1,6):
                if btc_data[j+1,0] > (1+threshold*th)*btc_data[j,0]: # price increase by 0.5%
                    y_oh[0,th-1,0]=1
                elif btc_data[j+1,0] < (1-threshold*th)*btc_data[j,0]: # price decrease by 0.5%
                    y_oh[0,th-1,2]=1
                else:
                    y_oh[0,th-1,1]=1
                if btc_data[j+2,0] > (1+threshold*th)*btc_data[j,0]: # price increase by 0.5%
                    y_oh[1,th-1,0]=1
                elif btc_data[j+2,0] < (1-threshold*th)*btc_data[j,0]: # price decrease by 0.5%
                    y_oh[1,th-1,2]=1
                else:
                    y_oh[1,th-1,1]=1
                if btc_data[j+3,0] > (1+threshold*th)*btc_data[j,0]: # price increase by 0.5%
                    y_oh[2,th-1,0]=1
                elif btc_data[j+3,0] < (1-threshold*th)*btc_data[j,0]: # price decrease by 0.5%
                    y_oh[2,th-1,2]=1
                else:
                    y_oh[2,th-1,1]=1
                if btc_data[j+5,0] > (1+threshold*th)*btc_data[j,0]: # price increase by 0.5%
                    y_oh[3,th-1,0]=1
                elif btc_data[j+5,0] < (1-threshold*th)*btc_data[j,0]: # price decrease by 0.5%
                    y_oh[3,th-1,2]=1
                else:
                    y_oh[3,th-1,1]=1
                if btc_data[j+10,0] > (1+threshold*th)*btc_data[j,0]: # price increase by 0.5%
                    y_oh[4,th-1,0]=1
                elif btc_data[j+10,0] < (1-threshold*th)*btc_data[j,0]: # price decrease by 0.5%
                    y_oh[4,th-1,2]=1
                else:
                    y_oh[4,th-1,1]=1

            Y = np.append(Y,y,axis=1)
            Y_oh = np.append(Y_oh,y_oh,axis=-1)

            tmp = np.zeros((21,1))
            tmp[0,0] = btc_data[j,0] # last price
            tmp[1,0] = btc_data[j,1] # volume (24h)
            tmp[2,0] = btc_data[j,2] # Bid
            tmp[3,0] = btc_data[j,3] # Ask
            mid_price = (order_log[j,0,2]+order_log[j,0,5])/2
            for k in range (0,500):
                tmp[4,0] = tmp[4,0] + order_log[j,k,0] #/(np.abs(order_log[j,k,2]-mid_price)+epsilon) # weight of total sell
                tmp[5,0] = tmp[5,0] + order_log[j,k,3] #/(np.abs(order_log[j,k,5]-mid_price)+epsilon) # weight of total buy
            k=0
            while order_log[j,k,2]<mid_price*1.005 and k<499:
                tmp[6,0] = tmp[6,0] + order_log[j,k,0] # total sell quantity within 0.5% from price
                k = k+1
            while order_log[j,k,2]<mid_price*1.01 and k<499:
                tmp[7,0] = tmp[7,0] + order_log[j,k,0] # total sell quantity between  0.5% to 1% from price
                k = k+1
            while order_log[j,k,2]<mid_price*1.02 and k<499:
                tmp[8,0] = tmp[8,0] + order_log[j,k,0] # total sell quantity between 1% to 2% from price
                k = k+1
            while order_log[j,k,2]<mid_price*1.05 and k<499:
                tmp[9,0] = tmp[9,0] + order_log[j,k,0] # total sell quantity between 2% to 5% from price
                k = k+1
            while order_log[j,k,2]<mid_price*1.1 and k<499:
                tmp[10,0] = tmp[10,0] + order_log[j,k,0] # total sell quantity between 5% to 10% from price
                k = k+1
            k=0
            while order_log[j,k,5]>mid_price*0.995 and k<499:
                tmp[11,0] = tmp[11,0] + order_log[j,k,3] # total buy quantity within 0.5% from price
                k = k+1
            while order_log[j,k,5]>mid_price*0.99 and k<499:
                tmp[12,0] = tmp[12,0] + order_log[j,k,3] # total buy quantity between 0.5% to 1% from price
                k = k+1
            while order_log[j,k,5]>mid_price*0.98 and k<499:
                tmp[13,0] = tmp[13,0] + order_log[j,k,3] # total buy quantity between 1% to 2% from price
                k = k+1
            while order_log[j,k,5]>mid_price*0.95 and k<499:
                tmp[14,0] = tmp[14,0] + order_log[j,k,3] # total buy quantity between 2% to 5% from price
                k = k+1
            while order_log[j,k,5]>mid_price*0.9 and k<499:
                tmp[15,0] = tmp[15,0] + order_log[j,k,3] # total buy quantity between 5% to 10% from price
                k = k+1
            tmp[16,0] = history[j,1]
            tmp[17,0] = history[j,2]
            tmp[18,0] = history[j,3]
            tmp[19,0] = history[j,4]
            tmp[20,0] = history[j,5]
            B = np.append(B,tmp,axis=1)
    print (np.shape(Y))
    print (np.shape(X))
    Y = Y[:,1:-1]
    X = X[:,1:-1]
    B = B[:,1:-1]
    print (B[4,1],B[5,1])
#    plt.plot(order_log[1,:,2],order_log[1,:,1])
#    plt.plot(order_log[1,:,5],order_log[1,:,4])
#    plt.subplot(2,1,1)
#    plt.plot(B[4,:])
#    plt.plot(B[5,:])
#    plt.plot(B[0,:]-7000)
#    plt.subplot(2,1,2)
#    plt.plot(B[19,:])
#    plt.plot(B[17,:])
#    plt.plot(B[0,:]-8000)
#    plt.show()    
    np.save('../dataset/X_new_100',X)
    np.save('../dataset/Y_new_100',Y)
    np.save('../dataset/Y_oh_new_100',Y_oh)
    np.save('../dataset/B_new_100',B)
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

def plot_B(B):
#    plt.plot(B[4,:])
#    plt.plot(B[5,:])
    plt.plot((B[5,:]-B[4,:])*10+7500)
    plt.plot(B[0,:])
    plt.show()


if __name__ == '__main__':
    order_100()
    exit()
#    plot_data()
#    exit()
    keep_prob = 1
    B = np.load('../dataset/B_100.npy')
    num_features = np.shape(B)[0]
#    plot_B(B)
#    exit()

