# coding: utf-8
# this program is preview minst dataset as image

# import library
import numpy as np
import pickle

# other library can not use.

# load minst dataset
# dataset dir is ./data/
# dataset file is t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte, train-images-idx3-ubyte, train-labels-idx1-ubyte

# load image data
def load_image(file_path):
    with open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data

# load label data
def load_label(file_path):
    with open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    return T

# load dataset
def load_dataset(normalize=True, flatten=True, one_hot_label=False):
    train_images = load_image('data/train-images-idx3-ubyte')
    train_labels = load_label('data/train-labels-idx1-ubyte')
    test_images = load_image('data/t10k-images-idx3-ubyte')
    test_labels = load_label('data/t10k-labels-idx1-ubyte')
    
    if normalize:
        train_images = train_images / 255.0
        test_images = test_images / 255.0
    
    if flatten:
        train_images = train_images.reshape(-1, 784)
        test_images = test_images.reshape(-1, 784)
    else:
        train_images = train_images.reshape(-1, 1, 28, 28)
        test_images = test_images.reshape(-1, 1, 28, 28)
    
    if one_hot_label:
        train_labels = _change_one_hot_label(train_labels)
        test_labels = _change_one_hot_label(test_labels)
    
    return train_images, train_labels, test_images, test_labels

def save_image(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)