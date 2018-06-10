import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

num_epochs = 1000
batch_size = 128
num_steps = 20  # Limits the back propagation steps and makes the current prediction dependent on these many inputs
num_hidden_units = 10  # Number of Neurons
learning_rate = 0.0001
activation_fn = tf.nn.tanh
data_gather_steps = 10


def gen_data():
    x_tmp = np.load('../dataset/X_100_moredata.npy')
    y_tmp = np.load('../dataset/Y_100_moredata.npy')
    x = x_tmp
    y = np.zeros((1, np.shape(y_tmp)[1]))
    y[0, :] = y_tmp[0, :] # [0,:] for 1min, [1,:] for 2min, [2,:] for 3min, [3,:] for 5min, [4,:] for 10min
    print(np.shape(x))
    print(np.shape(y))
    return x, y


def gen_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
    num_features_x = raw_x.shape[0]
    num_samples = raw_x.shape[1]
    assert num_samples == raw_y.shape[1]
    assert num_features_x % num_steps == 0

    extra_iteration = 0
    if num_samples % batch_size != 0:
        extra_iteration = 0

    for i in range(int(num_samples // batch_size) + extra_iteration):
        if ((i + 1) * batch_size) < num_samples:
            x_to_return_tmp = raw_x[:, i * batch_size:(i+1) * batch_size]
            x_to_return_tmp_tr = x_to_return_tmp.T
            x_to_return = \
                x_to_return_tmp_tr.reshape((x_to_return_tmp_tr.shape[0],
                                            num_steps, int(x_to_return_tmp_tr.shape[1] / num_steps)))
            y_to_return = raw_y[:, i * batch_size:(i+1) * batch_size]
            y_to_return = y_to_return.T
        else:
            x_to_return_tmp = raw_x[:, i * batch_size:num_samples]
            x_to_return_tmp_tr = x_to_return_tmp.T
            x_to_return = \
                x_to_return_tmp_tr.reshape((x_to_return_tmp_tr.shape[0],
                                            num_steps, int(x_to_return_tmp_tr.shape[1] / num_steps)))
            y_to_return = raw_y[:, i * batch_size:num_samples]
            y_to_return = y_to_return.T
        # print(x_to_return.shape)
        # print(y_to_return.shape)
        yield (x_to_return, y_to_return)


def gen_epochs(data, batch_size, num_epochs, num_steps):
    for i in range(num_epochs):
        yield gen_batch(data, batch_size, num_steps)


# Generate data
generated_data = gen_data()
(generated_data_X, generated_data_Y) = generated_data
m = generated_data_X.shape[1]
print("m is " + str(m))
train_set_count = int(m * 0.9)
dev_set_count = m - train_set_count
train_set = (generated_data_X[:, 0:train_set_count], generated_data_Y[:, 0:train_set_count])
(tr_set_x, tr_set_y) = train_set
print("tr set" + str(tr_set_x.shape))
dev_set = (generated_data_X[:, train_set_count:m], generated_data_Y[:, train_set_count:m])
(dev_set_x, dev_set_y) = dev_set
print("dev set" + str(dev_set_x.shape))

assert generated_data_X.shape[0] % num_steps == 0
input_size_per_time_step = int(generated_data_X.shape[0] / num_steps)

# Create Place Holders
x = tf.placeholder(tf.float32, [None, num_steps, input_size_per_time_step])
y = tf.placeholder(tf.float32, [None, generated_data_Y.shape[0]])
# init_state = (tf.zeros([batch_size, num_hidden_units]), tf.zeros([batch_size, num_hidden_units]))  # Initial Activations
# init_state = tf.placeholder(tf.float32, [None, num_hidden_units])

# Inputs - Unstack to get a list of 'num_steps' tensors of shape (batch_size, input_size_per_time_step)
rnn_inputs = tf.unstack(x, num_steps, axis=1)

# RNN
rnn_cell = tf.nn.rnn_cell.LSTMCell(num_hidden_units, activation=activation_fn)
# all_activations, final_activation = tf.nn.static_rnn(rnn_cell, rnn_inputs, initial_state=init_state, dtype=tf.float32)
all_activations, final_activation = tf.nn.static_rnn(rnn_cell, rnn_inputs, dtype=tf.float32)

# Create the weights and biases
# Prediction is a Sigmoid, hence 1
W = tf.get_variable('W', [num_hidden_units, 1], tf.float32, tf.initializers.random_normal)
b = tf.get_variable('b', [1, 1], tf.float32, tf.initializers.zeros)

# Forward Propagation
logits = tf.matmul(all_activations[-1], W) + b
required_prediction = tf.nn.sigmoid(logits)

# tv = tf.trainable_variables()
# regularization_cost = tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])

losses = [tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y)]
total_loss = tf.reduce_mean(losses) # + regularization_cost
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)
# train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)


def train_network(train_data, dev_data, batch_size, num_epochs, num_steps, num_hidden_units, verbose=True):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        predictions_sigmoid = []
        train_accuracy_all = []
        dev_accuracy_all = []
        for idx, epoch in enumerate(gen_epochs(train_data, batch_size, num_epochs, num_steps)):
            training_loss = 0
            for step, (X_data, Y_data) in enumerate(epoch):
                # print(step)
                x_step_data = X_data
                y_step_data = Y_data
                tr_losses, training_loss_, training_state, _, prediction_last = \
                    sess.run([losses,
                              total_loss,
                              final_activation,
                              train_step,
                              required_prediction],
                             feed_dict={x: x_step_data, y: y_step_data}) # , init_state: training_zero_state})
                training_loss += training_loss_
            training_losses.append(training_loss_)
            if idx % data_gather_steps == 0:
                if verbose:
                    print("Loss at epoch " + str(idx) + " = " + str(training_loss_))

                t_acc, y_data1_flatten_all_t, predictions_epoch_all_t = calculate_accuracy(train_data, sess, "Train")
                train_accuracy_all.append(t_acc)
                d_acc, y_data1_flatten_all_d, predictions_epoch_all_d = calculate_accuracy(dev_data, sess, "Dev")
                dev_accuracy_all.append(d_acc)
                predictions_sigmoid.clear()

    return training_losses, train_accuracy_all, dev_accuracy_all


def calculate_accuracy(data, sess, data_type):
    correct_predictions = 0
    total_predictions = 0
    y_all = []
    pred_all = []
    for idx1, epoch1 in enumerate(gen_epochs(data, batch_size=batch_size, num_steps=num_steps, num_epochs=1)):
        # print("inside0")
        for step1, (X_data1, Y_data1) in enumerate(epoch1):
            x_step_data = X_data1
            y_step_data = Y_data1
            pred_res = sess.run([required_prediction], feed_dict={x: x_step_data, y: y_step_data})
            predictions_sigmoid_array1 = np.array(pred_res)
            predictions_flatten = predictions_sigmoid_array1.flatten()
            predictions_flatten_rounded = np.round(predictions_flatten)
            y_data1_flatten = Y_data1.flatten()
            y_all.append(y_data1_flatten)
            pred_all.append(predictions_flatten_rounded)
            correct_predictions_vector = np.equal(predictions_flatten_rounded, y_data1_flatten)
            total_predictions += y_data1_flatten.shape[0]
            correct_predictions += np.sum(correct_predictions_vector)

    accuracy = correct_predictions / total_predictions
    print(data_type + " Accuracy: " + str(accuracy))
    return accuracy,y_all,pred_all


def plot_data(costs, train_accuracy_all, dev_accuracy_all):
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    # Plot the accuracy
    plt.plot(np.squeeze(train_accuracy_all))
    plt.plot(np.squeeze(dev_accuracy_all))
    plt.ylabel('Accuracy')
    plt.xlabel('iterations')
    plt.legend(['Training Set Accuracy', 'Dev Set Accuracy'], loc='upper left')
    plt.show()


costs, train_accuracy_all, dev_accuracy_all = \
    train_network(train_set, dev_set, batch_size, num_epochs, num_steps, num_hidden_units)
plot_data(costs, train_accuracy_all, dev_accuracy_all)






































