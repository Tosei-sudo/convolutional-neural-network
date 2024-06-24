# coding: utf-8

# 必要なライブラリをインポート
from data import dataset
import numpy as np
from models.ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork
from models.optimizer import Adam

# データセットの読み込み
x_train, t_train, x_test, t_test = dataset.load_dataset(flatten=False, one_hot_label=True)

# ハイパーパラメータ
epochs = 10
batch_size = 100
learning_rate = 0.01
train_size = x_train.shape[0]
iters_num = 10000

# モデルのインスタンス化
network = ConvolutionalNeuralNetwork(input_dim=(1, 28, 28), 
                                   conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                                   hidden_size=100, output_size=10)
optimizer = Adam()

train_loss_list = []
train_acc_list = []
test_acc_list = []
iter_per_epoch = max(train_size / batch_size, 1)

# トレーニングループ
for i in range(iters_num):
    
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    grads = network.gradient(x_batch, t_batch)
    
    optimizer.update(network.params, grads)
    
    loss = network.loss(x_batch, t_batch)
    print i, loss
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        pass
        # train_acc = network.accuracy(x_train, t_train)
        # train_acc_list.append(train_acc)
        
        # test_acc = network.accuracy(x_test, t_test)
        # test_acc_list.append(test_acc)
        
        # print "train acc, test acc | " + str(train_acc) + ", " + str(test_acc)

network.save_params("params.pkl")

import matplotlib.pylab as plt

# plt.plot(train_loss_list)
plt.plot(train_acc_list)
plt.plot(test_acc_list)
plt.show()