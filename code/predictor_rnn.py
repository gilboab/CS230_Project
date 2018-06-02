#!/usr/bin/env python
# from bittrex import bittrex
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

num_epochs = 10000
num_steps = 20  # Limits the back propagation steps and makes the current prediction dependent on these many inputs
batch_size = 128
num_hidden_units = 150  # Number of Neurons
learning_rate = 0.005


def gen_data():
    x = np.load('../dataset/X_100.npy')
    y = np.load('../dataset/Y_100.npy')
    print(np.shape(x))
    print(np.shape(y))
    return x, y


def gen_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
    num_features_x = raw_x.shape[0]
    num_samples = raw_x.shape[1]
    num_features_y = raw_y.shape[0]
    assert num_samples == raw_y.shape[1]

    batch_partition_length = num_samples // batch_size
    data_x = np.zeros((batch_size, num_features_x, batch_partition_length), dtype=np.float32)
    data_y = np.zeros((batch_size, num_features_y, batch_partition_length), dtype=np.float32)
    for i in range(batch_size):
        data_x[i] = raw_x[:, batch_partition_length * i:batch_partition_length * (i+1)]
        data_y[i] = raw_y[:, batch_partition_length * i:batch_partition_length * (i + 1)]

    # print("data_x shape: " + str(data_x.shape))
    # print("data_y shape: " + str(data_y.shape))

    epoch_size = batch_partition_length // num_steps
    for i in range(epoch_size):
        x_d = data_x[:, :, i * num_steps:(i + 1) * num_steps]
        y_d = data_y[:, :, i * num_steps:(i + 1) * num_steps]

        # print("x_d shape: " + str(x_d.shape))
        # print("y_d shape: " + str(y_d.shape))

        # Take the Transpose to shift the feature vector to the end and bring the steps component earlier
        x_final = np.transpose(x_d, [0, 2, 1])
        y_final = np.transpose(y_d, [0, 2, 1])
        yield (x_final, y_final)


def gen_epochs(data, num_epochs, num_steps):
    for i in range(num_epochs):
        yield gen_batch(data, batch_size, num_steps)


# Generate data
generated_data = gen_data()
(generated_data_X, generated_data_Y) = generated_data
m = generated_data_X.shape[1]
print("m is " + str(m))
train_set_count = int(m * 0.8)
dev_set_count = m - train_set_count
train_set = (generated_data_X[:, 0:train_set_count], generated_data_Y[:, 0:train_set_count])
(tr_set_x, tr_set_y) = train_set
print("tr set" + str(tr_set_x.shape))
dev_set = (generated_data_X[:, train_set_count:m], generated_data_Y[:, train_set_count:m])
(dev_set_x, dev_set_y) = dev_set
print("dev set" + str(dev_set_x.shape))

# Create Place Holders
# x = tf.placeholder(tf.float32, [batch_size, num_steps, generated_data_X.shape[0]])
x = tf.placeholder(tf.float32, [None, num_steps, generated_data_X.shape[0]])
# y = tf.placeholder(tf.float32, [batch_size, num_steps, generated_data_Y.shape[0]])
y = tf.placeholder(tf.float32, [None, num_steps, generated_data_Y.shape[0]])
init_state = (tf.zeros([batch_size, num_hidden_units]), tf.zeros([batch_size, num_hidden_units]))  # Initial Activations
# init_state = tf.zeros([batch_size, num_hidden_units])  # Initial Activations

# Inputs
rnn_inputs = tf.unstack(x, num_steps, axis=1)

# RNN
rnn_cell = tf.nn.rnn_cell.LSTMCell(num_hidden_units)
# all_activations, final_activation = tf.nn.static_rnn(rnn_cell, x, initial_state=init_state, dtype=tf.float32)
all_activations, final_activation = tf.nn.static_rnn(rnn_cell, rnn_inputs, initial_state=init_state, dtype=tf.float32)

# with tf.variable_scope('sigmoid'):
# Create the weights and biases
# Prediction is a Sigmoid, hence 1
W = tf.get_variable('W', [num_hidden_units, 1], tf.float32, tf.initializers.random_normal)
b = tf.get_variable('b', [1, 1], tf.float32, tf.initializers.zeros)
# b = tf.get_variable('b', [1], tf.float32, tf.initializers.zeros)
# W = tf.get_variable('W', [1, num_hidden_units], tf.float32, tf.initializers.random_normal)
# b = tf.get_variable('b', [1, 1], tf.float32, tf.initializers.zeros)

# Forward Propagation
logits_series = [tf.matmul(activation, W) + b for activation in all_activations]
# logits_series = [tf.matmul(W, activation) + b for activation in all_activations]
prediction_series = [tf.nn.sigmoid(logits) for logits in logits_series]

train_labels = tf.unstack(y, num_steps, axis=1)

# Prepare Backward Propagation
losses = [tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,
                                                                                                          train_labels)]
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)


def train_network(train_data, dev_data, batch_size, num_epochs, num_steps, num_hidden_units, verbose=True):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        predictions_sigmoid = []
        # (train_data_X, train_data_Y) = train_data
        for idx, epoch in enumerate(gen_epochs(train_data, num_epochs, num_steps)):
            training_loss = 0
            training_zero_state = (np.zeros((batch_size, num_hidden_units)), np.zeros((batch_size, num_hidden_units)))
            # if(verbose):
            #   print("\nEpoch", idx)
            for step, (X_data, Y_data) in enumerate(epoch):
                y_step_data = Y_data
                # print(X_data.shape)
                # print(Y_data.shape)
                # print("I am inside the step")
                # sess.run(train_step, feed_dict={x: X_data, y: Y_data, init_state: training_state})
                # sess.run(prediction_series, feed_dict={x: X_data, y: Y_data, init_state: training_state})
                tr_losses, training_loss_, training_state, _, prediction_res = \
                    sess.run([losses,
                              total_loss,
                              final_activation,
                              train_step,
                              prediction_series],
                             feed_dict={x: X_data, y: Y_data, init_state: training_zero_state})
                # print("prediction_res = " + str(prediction_res))
                # print("tr_losses = " + str(tr_losses))
                # print("training_loss_ = " + str(training_loss_))
                # print("training_state = " + str(training_state))
                # print(step)
                training_loss += training_loss_
                #if step + 1 == num_steps:
                    # Last Step - Get Predictions
                predictions_sigmoid.append(prediction_res)
            training_losses.append(training_loss_)
            if idx % 10 == 0:
                if verbose:
                    print("Loss at epoch " + str(idx) + " = " + str(training_loss))
                # Calculate Training Accuracy
                predictions_sigmoid_array = np.array(predictions_sigmoid[-1])
                # print(predictions_sigmoid_array.shape)
                # print(y_step_data.shape)
                predictions = np.equal((np.round(np.transpose(predictions_sigmoid_array, [1, 0, 2])).flatten()),
                                       y_step_data.flatten())
                train_accuracy = np.sum(predictions) / predictions.shape[0]
                print("Train Accuracy: " + str(train_accuracy))
                predictions_sigmoid.clear()
                # print("predictions shape = " + str(predictions.shape[0]))
                # print(Y_data.shape)
                # accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
                # print(predictions)

                # Dev Accuracy
                # print("Time for dev accuracy")
                for idx1, epoch1 in enumerate(gen_epochs(dev_data, 1, num_steps)):
                    # print("inside0")
                    for step1, (X_data1, Y_data1) in enumerate(epoch1):
                        # print("inside")
                        dev_pred_res = sess.run([prediction_series], feed_dict={x: X_data1, y: Y_data1})
                        predictions_sigmoid_array1 = np.array(dev_pred_res[0])
                        # print(predictions_sigmoid_array1.shape)
                        # print(Y_data1.shape)
                        predictions1 = np.equal((np.round(np.transpose(predictions_sigmoid_array1, [1, 0, 2])).flatten()),
                                              Y_data1.flatten())
                        dev_accuracy = np.sum(predictions1) / predictions1.shape[0]
                        print("Dev Accuracy: " + str(dev_accuracy))

    return training_losses


training_losses = train_network(train_set, dev_set, batch_size, num_epochs, num_steps, num_hidden_units)
# print(training_losses)








































