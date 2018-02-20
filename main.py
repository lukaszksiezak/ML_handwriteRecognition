import scipy.io
import numpy as np

if __name__ == "__main__":
    
    input_layer_size  = 400;  # 20x20 Input Images of Digits
    hidden_layer_size = 25;   # 25 hidden units
    num_labels = 10;          # 10 labels, from 1 to 10   
    mat = scipy.io.loadmat('handwrite_recognition_python\images_data.mat')

    
