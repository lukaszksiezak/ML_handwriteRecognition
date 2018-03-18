import scipy.io
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split


def learn_nn_network(X_train, X_test, y_train, y_test, batches):

    learning_rate = 0.5
    epochs = 10

    # input x - for 20 x 20 pixels = 400
    x = tf.placeholder(tf.float32, [None, 400])
    #  output data placeholder - 10 digits
    y = tf.placeholder(tf.float32, [None, 1])

    # weights connecting the input to the hidden layer (100 nodes in hidden
    # layer)
    W1 = tf.Variable(tf.random_normal([400, 100], stddev=0.03), name='W1')
    b1 = tf.Variable(tf.random_normal([100]), name='b1')

    # weights connecting the hidden layer to the output layer
    W2 = tf.Variable(tf.random_normal([100, 10], stddev=0.03), name='W2')
    b2 = tf.Variable(tf.random_normal([10]), name='b2')

    hidden_out = tf.add(tf.matmul(x, W1), b1)
    hidden_out = tf.nn.relu(hidden_out)

    # hidden layer output - in this case, let's use a softmax
    # activated output layer
    y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

    # operation converting the output y_ to a clipped version,
    # limited between 1e-10 to 0.999999. (Avoiding log(0))
    y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
    cross_entropy = -tf.reduce_mean(
        tf.reduce_sum(
            y * tf.log(
                y_clipped) + (1 - y) * tf.log(
                    1 - y_clipped), axis=1))

    optimiser = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate).minimize(cross_entropy)

    # setup the initialisation operator
    init_op = tf.global_variables_initializer()

    # define an accuracy assessment operation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:        
        sess.run(init_op)
        for epoch in range(epochs):
            avg_cost = 0
            batch_x_train = np.array(X_train).reshape(batches, 400)
            batch_y_train = np.array(y_train).reshape(batches, 1)
            for i in range(batches):
                _, c = sess.run(
                    [optimiser, cross_entropy], feed_dict={
                        x: batch_x_train, y: batch_y_train})
                avg_cost += c / batches
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(
                avg_cost))
        batch_x_test = np.array(X_test).reshape(batches, 400)
        batch_y_test = np.array(y_test).reshape(batches, 1)
        print(sess.run(
            accuracy, feed_dict={x: batch_x_test, y: batch_y_test}))
    return sess


def mirror_arr(arr_data):
    for i in range(len(arr_data)):
        arr_data[i] = arr_data[i][::-1]

    arr_data = np.rot90(arr_data, axes=(-2, -1))
    return arr_data


def img_title(correct_val, pred_val):
    correct_val = correct_val[0]

    if correct_val == 10:
        correct_val = 0

    return "Correct val:%s; Predicted val:%s" % (correct_val, pred_val)


def display_image(data, title):
    plt.imshow(data, cmap='gray')
    plt.title(title)
    plt.show()


def prep_data_to_disp(data):
    arr_data = np.array(data)
    arr_data = np.reshape(arr_data, (20, 20))
    return mirror_arr(arr_data)


if __name__ == "__main__":

    # size_of_batch = 400  # 20x20 Input Images of Digits
    mat = scipy.io.loadmat('images_data.mat')

    X = mat['X']  # 5000x400 (5000 pictures of size 20x20)
    y = mat['y']

    # split test data into training & test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0)

    mixed_indices = np.random.permutation(len(X))

    number_of_batches = 3000  # 60% of total (5000)
    model = learn_nn_network(
        X_train, X_test, y_train, y_test, number_of_batches)

    # model verification against picture (demo purposes TODO)
    # for i in mixed_indices:
    #     predicted = 0  # TODO
    #     correct = y[i]
    #     display_image(prep_data_to_disp(X[i]), img_title(correct, predicted))
