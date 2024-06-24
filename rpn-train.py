# coding: utf-8
import numpy as np
from models.DeepNetwork import VGG16
from data import dataset
from models.optimizer import Adam
import pickle
import os

x_train, t_train, x_test, t_test = dataset.load_dataset(flatten=False, one_hot_label=True)

# 入力次元数とアンカー数を定義
input_dim = (1, 280, 280)  # 例えば、RGB画像
num_anchors = 10
epochs = 10
batch_size = 100
learning_rate = 0.01
train_size = x_train.shape[0]
iters_num = 10000

padding = 2
stride = 1

# x_train(60000, 1, 28, 28)とx_test(10000, 1, 28, 28)をinput_dim(n, 1, 112, 112)に拡大　中間は平均値を内挿
path = "./data/100100TrainImage.ndarray"
if os.path.exists(path):
    x_train = np.load(path)
else:
    x_train = np.repeat(np.repeat(x_train, 4, axis=2), 4, axis=3)
    # save
    np.save(path, x_train)
    
path = "./data/100100TestImage.ndarray"
if os.path.exists(path):
    x_test = np.load(path)
else:
    x_test = np.repeat(np.repeat(x_test, 4, axis=2), 4, axis=3)
    np.save(path, x_test)
print x_train.shape, x_test.shape

# RPNモデルのインスタンスを作成
network = VGG16(input_dim=input_dim, output_size=num_anchors, padding=padding, stride=stride)
optimizer = Adam()

# # 例として、ランダムな入力データを作成

train_loss_list = []
train_acc_list = []
test_acc_list = []
iter_per_epoch = max(train_size / batch_size, 1)

# トレーニングループ
for i in range(iters_num):
    
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    print("start batch {0}".format(i))
    
    grads = network.gradient(x_batch, t_batch)
    print("calc gradient {0}".format(i))
    
    optimizer.update(network.params, grads)
    print("update params {0}".format(i))
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    print i, loss
    
    test_batch_mask = np.random.choice(x_test.shape[0], batch_size)
    test_acc = network.accuracy(x_test[test_batch_mask], t_test[test_batch_mask])
    print test_acc
    
    if i % iter_per_epoch == 0:
        pass
        # train_acc = network.accuracy(x_train, t_train)
        # train_acc_list.append(train_acc)
        
        
        # print "train acc, test acc | " + str(train_acc) + ", " + str(test_acc)

network.save_params("vgg16_params.pkl")

test_acc = network.accuracy(x_test, t_test)
test_acc_list.append(test_acc)

print "train acc, test acc | " + str(test_acc)