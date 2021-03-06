import scipy.io
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split


def _prepare_test_labels(idx):
    lbl_container = np.zeros((10,), dtype=int)
    if(idx == 10):
        idx = 0
    lbl_container[idx] = 1
    return lbl_container


def _next_image(Xtrain, Ytrain, idx):
    im = np.array(Xtrain[idx][:]).reshape(1, 400)
    lbl_container = _prepare_test_labels(Ytrain[idx][0])
    return im, lbl_container.reshape(1, 10)


def neural_net(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer


def learn_nn_network(X_train, X_test, y_train, y_test, batches):

    # Parameters
    learning_rate = 0.01
    n_hidden_1 = 25  # 1st layer number of neurons

    num_input = 400
    num_classes = 10  # total classes (0-9 digits)

    X = tf.placeholder("float", [None, num_input])
    Y = tf.placeholder("float", [None, num_classes])

    weights = {
        'h1': tf.Variable(
            tf.random_normal([num_input, n_hidden_1]), name="theta1"),
        'out': tf.Variable(
            tf.random_normal([n_hidden_1, num_classes]), name="theta2")
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1]), name="b1"),
        'out': tf.Variable(tf.random_normal([num_classes]), name="b2")
    }

    logits = neural_net(X, weights, biases)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init_op = tf.global_variables_initializer()
    
    perm = np.arange(batches)
    np.random.shuffle(perm)

    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(batches):
            batch_x_train, batch_y_train = _next_image(
                X_train, y_train, perm[i])

            sess.run(train_op, feed_dict={
                X: batch_x_train, Y: batch_y_train})

        batch_x_test = np.array(X_test).reshape(1000, 400)
        batch_y_test = np.array(y_test).reshape(1000, 10)
        print(sess.run(
            accuracy, feed_dict={X: batch_x_test, Y: batch_y_test}))

        saver = tf.train.Saver()
        save_path = saver.save(sess, "/model/model.ckpt")
        print("Model saved in file: %s" % save_path)

    return sess


def mirror_arr(arr_data):
    for i in range(0, len(arr_data)):
        arr_data[i] = arr_data[i][::-1]

    arr_data = np.rot90(arr_data, axes=(-2, -1))
    return arr_data


def img_title(correct_val, pred_val):
    correct_val = correct_val[0]

    if correct_val == 10:
        correct_val = 0

    return "Correct val:%s; Predicted val:%s" % (correct_val, pred_val[0])


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
    X_train, X_test, y_train, _y_test = train_test_split(
        X, y, test_size=0.2)

    y_test = np.zeros((1000, 10), dtype=int)
    for i in range(0, len(y_test)):
        y_test[i] = _prepare_test_labels(_y_test[i][0])

    mixed_indices = np.random.permutation(len(X))
    number_of_batches = 4000  # 80% of total (5000)
    model = learn_nn_network(
        X_train, X_test, y_train, y_test, number_of_batches)

    tf.reset_default_graph()
    saver = tf.train.import_meta_graph("/model/model.ckpt.meta")

    with tf.Session() as sess:
        saver.restore(sess,  tf.train.latest_checkpoint('/model/'))
        graph = tf.get_default_graph()
        theta1 = graph.get_tensor_by_name("theta1:0")
        theta2 = graph.get_tensor_by_name("theta2:0")
        b1 = graph.get_tensor_by_name("b1:0")
        b2 = graph.get_tensor_by_name("b2:0")

        weights = {
            'h1': tf.Variable(theta1),
            'out': tf.Variable(theta2)
        }
        biases = {
            'b1': tf.Variable(b1),
            'out': tf.Variable(b2)
        }
        
        init_op = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        sess.run(init_op)
        sess.run(init_l)
        print("Model restored.")

        # Fun with prediction:
        for i in mixed_indices:
            Xi = np.array(X[i][:]).reshape(1, 400)
            logits = neural_net(np.float32(Xi), weights, biases)
            prediction = tf.nn.softmax(logits)
            correct_pred = tf.argmax(prediction, 1)
            predicted = sess.run(correct_pred)
            correct = y[i]

            display_image(
                prep_data_to_disp(X[i]), img_title(correct, predicted))

