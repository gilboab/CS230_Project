import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

num_epochs = 100
num_steps = 1  # Limits the back propagation steps and makes the current prediction dependent on these many inputs
batch_size = 128
num_hidden_units = 150  # Number of Neurons
learning_rate = 0.0005
data_gather_steps = 40


def gen_data():
    x = np.load('../dataset/X_100_moredata.npy')
    y_tmp = np.load('../dataset/Y_100_moredata.npy')
    y = np.zeros((1, np.shape(y_tmp)[1]))
    y[0, :] = y_tmp[4, :] # [0,:] for 1min, [1,:] for 2min, [2,:] for 3min, [3,:] for 5min, [4,:] for 10min
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

        # print("x_final shape: " + str(x_final.shape))
        # print("y_final shape: " + str(y_final.shape))

        yield (x_final, y_final)


def gen_epochs(data, batch_size, num_epochs, num_steps):
    for i in range(num_epochs):
        yield gen_batch(data, batch_size, num_steps)


# Generate data
generated_data = gen_data()
(generated_data_X, generated_data_Y) = generated_data
m = generated_data_X.shape[1]
print("m is " + str(m))
train_set_count = int(m * 0.90)
dev_set_count = m - train_set_count
train_set = (generated_data_X[:, 0:train_set_count], generated_data_Y[:, 0:train_set_count])
(tr_set_x, tr_set_y) = train_set
print("tr set" + str(tr_set_x.shape))
dev_set = (generated_data_X[:, train_set_count:m], generated_data_Y[:, train_set_count:m])
(dev_set_x, dev_set_y) = dev_set
print("dev set" + str(dev_set_x.shape))

# Create Place Holders
x = tf.placeholder(tf.float32, [None, num_steps, generated_data_X.shape[0]])
y = tf.placeholder(tf.float32, [None, num_steps, generated_data_Y.shape[0]])
# init_state = tf.placeholder(tf.float32, [None, num_hidden_units])  # Initial Activations

# Inputs
rnn_inputs = x

# RNN
rnn_cell = tf.nn.rnn_cell.LSTMCell(num_hidden_units)
all_activations, final_activation = tf.nn.dynamic_rnn(rnn_cell, rnn_inputs, dtype=tf.float32)

with tf.variable_scope('sigmoidw'):
    # Create the weights and biases
    # Prediction is a Sigmoid, hence 1
    W = tf.get_variable('W', [num_hidden_units, 1], tf.float32, tf.initializers.random_normal)
with tf.variable_scope('sigmoidb'):
    b = tf.get_variable('b', [1, 1], tf.float32, tf.initializers.zeros)

# Forward Propagation
logits_series = tf.map_fn(lambda u: tf.matmul(u, W) + b, all_activations)
prediction_series = tf.map_fn(lambda v: tf.nn.sigmoid(v), logits_series)

# train_labels = tf.unstack(y, num_steps, axis=1)
train_labels = y

# Prepare Backward Propagation
tv = tf.trainable_variables(scope='sigmoidw')
# tv = tf.trainable_variables()
regularization_cost = tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])

losses = [tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_series, labels=train_labels)]
total_loss = tf.reduce_mean(losses) + regularization_cost
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)


def train_network(train_data, dev_data, batch_size, num_epochs, num_steps, num_hidden_units, verbose=True):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        predictions_sigmoid = []
        all_y_data = []
        train_accuracy_all = []
        dev_accuracy_all = []
        y_data1_flatten_all_t = []
        predictions_epoch_all_t = []
        y_data1_flatten_all_d = []
        predictions_epoch_all_d = []
        for idx, epoch in enumerate(gen_epochs(train_data, batch_size, num_epochs, num_steps)):
            training_loss = 0
            for step, (X_data, Y_data) in enumerate(epoch):
                y_step_data = Y_data
                tr_losses, training_loss_, training_state, _, logits_res, prediction_res = \
                    sess.run([losses,
                              total_loss,
                              final_activation,
                              train_step,
                              logits_series,
                              prediction_series],
                             feed_dict={x: X_data, y: Y_data}) #, init_state: training_zero_state})
                training_loss += training_loss_
                predictions_sigmoid.append(prediction_res)
                all_y_data.append(y_step_data)
                # print((np.array(prediction_res)).shape)
            training_losses.append(training_loss_)
            if idx % data_gather_steps == 0:
                if verbose:
                    print("Loss at epoch " + str(idx) + " = " + str(training_loss))
                # Calculate Training Accuracy
                t_acc, y_data1_flatten_all_t, predictions_epoch_all_t = calculate_accuracy(train_data, sess, "Train")
                train_accuracy_all.append(t_acc)
                d_acc, y_data1_flatten_all_d, predictions_epoch_all_d = calculate_accuracy(dev_data, sess, "Dev")
                dev_accuracy_all.append(d_acc)
                predictions_sigmoid.clear()

    return training_losses, train_accuracy_all, dev_accuracy_all,y_data1_flatten_all_t,\
        predictions_epoch_all_t,y_data1_flatten_all_d,predictions_epoch_all_d


def calculate_accuracy(data, sess, data_type):
    correct_predictions = 0
    total_predictions = 0
    y_all = []
    pred_all = []
    for idx1, epoch1 in enumerate(gen_epochs(data, batch_size=128, num_steps=1, num_epochs=1)):
        # print("inside0")
        for step1, (X_data1, Y_data1) in enumerate(epoch1):
            pred_res = sess.run([prediction_series], feed_dict={x: X_data1, y: Y_data1})
            predictions_sigmoid_array1 = np.array(pred_res[0])
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
    plt.xlabel('iterations (per 50)')
    plt.legend(['Training Set Accuracy', 'Dev Set Accuracy'], loc='upper left')
    plt.show()


def get_total_predictions_table(y_set, prediction_set):
    print(y_set.shape)
    print(prediction_set.shape)
    y_set_flatten = y_set.flatten()
    prediction_set_flatten = prediction_set.flatten()
    increase_count_set = np.sum(y_set_flatten)
    decrease_same_count_set = y_set_flatten.shape[0] - increase_count_set
    increase_count_set_prediction = np.sum(prediction_set_flatten)
    decrease_same_count_set_prediction = prediction_set_flatten.shape[0] - increase_count_set_prediction
    print("Increase_count_set = " + str(increase_count_set))
    print("decrease_same_count_set = " + str(decrease_same_count_set))
    print("increase_count_set_prediction = " + str(increase_count_set_prediction))
    print("decrease_same_count_set_prediction = " + str(decrease_same_count_set_prediction))


costs, train_accuracy_all, dev_accuracy_all, y_data1_flatten_all_t,predictions_epoch_all_t, \
    y_data1_flatten_all_d,predictions_epoch_all_d = \
    train_network(train_set, dev_set, batch_size, num_epochs, num_steps, num_hidden_units)
plot_data(costs, train_accuracy_all, dev_accuracy_all)






































