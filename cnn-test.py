# coding: utf-8

# 必要なライブラリをインポート
from data import dataset
from models.ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork

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

network.load_params("params.pkl")

train_acc = network.accuracy(x_train, t_train)

test_acc = network.accuracy(x_test, t_test)

print "train acc, test acc | " + str(train_acc) + ", " + str(test_acc)
