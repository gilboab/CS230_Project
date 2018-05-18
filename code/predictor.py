#!/usr/bin/env python
#from bittrex import bittrex
import numpy as np
import datetime
import time
import csv
import matplotlib.pyplot as plt
from numpy import genfromtxt
import math
import h5py
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

def compute_cost(Z, Y, parameters, lamda = 0.1):
    """
    Computes the cost
    Arguments:
    Z -- output of forward propagation (output of the last LINEAR unit), of shape (3, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z
    
    Returns:
    cost - Tensor of the cost function
    """
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']
    W5 = parameters['W5']
    W6 = parameters['W6']
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z)
    labels = tf.transpose(Y)
    
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))
    #cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels) +
    #                     lamda*(tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4) + tf.nn.l2_loss(W5) + tf.nn.l2_loss(W6)))
    return cost



def initialize_parameters(L1,L2,L3,L4,L5, X_dim):
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [L1, X_dim]
                        b1 : [L1, 1]
                        W2 : [L2, L1]
                        b2 : [L2, 1]
                        W3 : [1, L2]
                        b3 : [1, 2]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3, W4, b4, W5, b5
    """
    
    W1 = tf.get_variable("W1", [L1,X_dim], initializer = tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [L1,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [L2,L1], initializer = tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [L2,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [L3,L2], initializer = tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [L3,1], initializer = tf.zeros_initializer())
    W4 = tf.get_variable("W4", [L4,L3], initializer = tf.contrib.layers.xavier_initializer())
    b4 = tf.get_variable("b4", [L4,1], initializer = tf.zeros_initializer())
    W5 = tf.get_variable("W5", [L5, L4], initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.get_variable("b5", [L5, 1], initializer=tf.zeros_initializer())
    W6 = tf.get_variable("W6", [1, L5], initializer=tf.contrib.layers.xavier_initializer())
    b6 = tf.get_variable("b6", [1, 1], initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,
                  "W5": W5,
                  "b5": b5,
                  "W6": W6,
                  "b6": b6}
    
    return parameters

def forward_propagation_np (X, parameters):
    """
    Implements the forward propagation for the model: 
    LINEAR -> RELU -> LINEAR --> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4", "W5", "b5"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    W5 = parameters['W5']
    b5 = parameters['b5']
    W6 = parameters['W6']
    b6 = parameters['b6']
    
    Z1 = np.dot(W1,X) + b1
    A1 = np.maximum(Z1,0,Z1) # relu
    Z2 = np.dot(W2,A1) + b2
    A2 = np.maximum(Z2,0,Z2) # relu
    Z3 = np.dot(W3,A2) + b3
    A3 = np.maximum(Z3,0,Z3) # relu
    Z4 = np.dot(W4,A3) + b4
    A4 = np.maximum(Z4,0,Z4) # relu
    Z5 = np.dot(W5, A4) + b5
    A5 = np.maximum(Z5, 0, Z5)  # relu
    Z6 = np.dot(W6,A5) + b6
    return Z6

def forward_propagation(X, parameters, keep_prob = 1):
    """
    Implements the forward propagation for the model: 
    LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> RELU --> LINEAR --> RELU --> LINEAR --> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4", "W5", "b5"
                  the shapes are given in initialize_parameters

    Returns:
    Z2 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    W5 = parameters['W5']
    b5 = parameters['b5']
    W6 = parameters['W6']
    b6 = parameters['b6']
    
    Z1 = tf.add(tf.matmul(W1,X),b1)                                  # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
    D1 = tf.nn.dropout(A1, keep_prob)  # DROP-OUT here
    Z2 = tf.add(tf.matmul(W2,D1),b2)                                 # Z2 = np.dot(W2, A1) + b2
    A2 = tf.nn.relu(Z2)                                              # A2 = relu(Z2)
    D2 = tf.nn.dropout(A2, keep_prob)  # DROP-OUT here
    Z3 = tf.add(tf.matmul(W3,D2),b3)                                 # Z3 = np.dot(W3, A2) + b3
    A3 = tf.nn.relu(Z3)                                              # A2 = relu(Z2)
    D3 = tf.nn.dropout(A3, keep_prob)  # DROP-OUT here
    Z4 = tf.add(tf.matmul(W4,D3),b4)                                 # Z3 = np.dot(W3, A2) + b3
    A4 = tf.nn.relu(Z4)                                              # A2 = relu(Z2)
    D4 = tf.nn.dropout(A4, keep_prob)  # DROP-OUT here
    Z5 = tf.add(tf.matmul(W5, D4), b5)  # Z3 = np.dot(W3, A2) + b3
    A5 = tf.nn.relu(Z5)  # A2 = relu(Z2)
    D5 = tf.nn.dropout(A5, keep_prob)  # DROP-OUT here
    Z6 = tf.add(tf.matmul(W6,D5),b6)                                 # Z3 = np.dot(W3, A2) + b3
    return Z6


def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.005,
          num_epochs = 5000, minibatch_size = 128, print_cost = True):
    """
    Implements a Five-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (i nput size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    L1 = 50  # 30
    L2 = 30  # 20
    L3 = 20  # 10
    L4 = 10  # 5
    L5 = 5
    X_dim = 200
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    # tf.set_random_seed(1)                             # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    train_accuracy = []
    dev_accuracy = []
    
    # Create Placeholders of shape (n_x, n_y)
    X = tf.placeholder(tf.float32, shape=(n_x,None))
    Y = tf.placeholder(tf.float32, shape=(n_y,None))
    
    # Initialize parameters
    parameters = initialize_parameters(L1,L2,L3,L4,L5,X_dim)
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z6 = forward_propagation(X, parameters)
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z6, Y, parameters)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 50 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

            if epoch % 50 == 0:
                # print accuracy after each of the 50 epochs
                # Calculate the correct predictions
                correct_prediction = tf.equal(tf.round(tf.sigmoid(Z6)), Y)
                # Calculate accuracy on the test set
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

                train_acc = accuracy.eval({X: X_train, Y: Y_train})
                dev_acc = accuracy.eval({X: X_test, Y: Y_test})
                train_accuracy.append(train_acc)
                dev_accuracy.append(dev_acc)
                print("Train Accuracy:", train_acc)
                print("Test Accuracy:", dev_acc)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Plot the accuracy
        plt.plot(np.squeeze(train_accuracy))
        plt.plot(np.squeeze(dev_accuracy))
        plt.ylabel('Accuracy')
        plt.xlabel('iterations (per 50)')
        plt.legend(['Training Set Accuracy', 'Dev Set Accuracy'], loc='upper left')
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.round(tf.sigmoid(Z6)), Y)
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters

def order_100 ():
    X = np.zeros((200,1))
    Y = np.zeros((5,1))
    epsilon = 0.00000001
    dp = 10 # delta price for binig the data
    for i in range (1,8):
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
#    print (np.shape(Y))
#    print (np.shape(X))
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

def test_results (X,Y,parameters):
    x = np.zeros((np.shape(X)[0],1))
    results = np.zeros((2,2))
    for i in range (0,np.shape(Y)[1]):
        x[:,0] = X[:,i]
        Z6 = forward_propagation_np(x, parameters)
        prediction = sigmoid(Z6)
        if Y[0,i]==1:
            results[0,0] +=1
            if prediction>0.5:
                results[1,0] +=1
        elif Y[0,i]==0:
            results[0,1] +=1
            if prediction<0.5:
                results[1,1] +=1
    print (results)
    return ()

if __name__ == '__main__':
#    order_100()
#    exit()
#    plot_data()
#    exit()
    X = np.load('../dataset/X_100.npy')
    Y_tmp = np.load('../dataset/Y_100.npy')
    Y = np.zeros((1,np.shape(Y_tmp)[1]))
    Y[0,:] = Y_tmp[4,:] # [0,:] for 1min, [1,:] for 2min, [2,:] for 3min, [3,:] for 5min, [4,:] for 10min
    print (np.shape(X))
    print (np.shape(Y))
    m = np.shape(X)[1]
    train_size = int(np.floor(m*0.8))
    dev_size = m-train_size
    mu = np.zeros((np.shape(X)[0],1))
    sigma = np.zeros((np.shape(X)[0],1))
    mu[:,0] = np.sum(X,axis=1)/m
    sigma[:,0] = np.sum(X**2,axis=1)/m
    print (np.shape(mu))
    print (np.shape(sigma))
    #X_norm = (X-mu)/sigma
    X_norm = X # no normalization
    s = np.arange(m)
    np.random.shuffle(s)
    X_norm=X_norm[:,s]
    Y=Y[:,s]
    #Y_one_hot = convert_to_one_hot(Y.astype(int), 2)
    X_train = X_norm[:,0:train_size]
    #Y_train = Y_one_hot[:,0:train_size]
    Y_train = Y[:,0:train_size]
    X_dev = X_norm[:,train_size:m]
    #Y_dev = Y_one_hot[:,train_size:m]
    Y_dev = Y[:,train_size:m]
#    print (np.shape(Y_dev))
#    print (np.sum(Y_dev,axis=1))
    parameters = model(X_train, Y_train, X_dev, Y_dev)
    test_results (X_train,Y_train,parameters)
    test_results (X_dev,Y_dev,parameters)
