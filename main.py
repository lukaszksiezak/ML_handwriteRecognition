import scipy.io
import numpy as np
from matplotlib import pyplot as plt


def mirror_arr(arr_data):
    for i in range(len(arr_data)):
        arr_data[i] = arr_data[i][::-1]

    arr_data = np.rot90(arr_data, axes=(-2, -1))
    return arr_data


def display_image(data):
    arr_data = np.array(data)
    arr_data = np.reshape(arr_data, (20, 20))
    plt.imshow(mirror_arr(arr_data), cmap='gray')
    plt.show()


if __name__ == "__main__":

    input_layer_size = 400  # 20x20 Input Images of Digits
    hidden_layer_size = 25   # 25 hidden units
    num_labels = 10          # 10 labels, from 1 to 10
    mat = scipy.io.loadmat('images_data.mat')

    X = mat['X']  # 5000x400 (5000 pictures of size 20x20)
    y = mat['y']

    mixed_indices = np.random.permutation(len(X))

    for i in mixed_indices:
        display_image(X[i])  # randomly choose picture

