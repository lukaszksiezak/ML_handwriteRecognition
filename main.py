import scipy.io
import numpy as np
from matplotlib import pyplot as plt


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

    input_layer_size = 400  # 20x20 Input Images of Digits
    hidden_layer_size = 25   # 25 hidden units
    num_labels = 10          # 10 labels, from 1 to 10
    mat = scipy.io.loadmat('images_data.mat')

    X = mat['X']  # 5000x400 (5000 pictures of size 20x20)
    y = mat['y']

    mixed_indices = np.random.permutation(len(X))

    for i in mixed_indices:
        predicted = 0  # TODO
        correct = y[i]
        display_image(prep_data_to_disp(X[i]), img_title(correct, predicted))
