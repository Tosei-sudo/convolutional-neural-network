# coding: utf-8

from dataset.dummy import File
from dataset import bmp
from dataset.functions import rgb2grayscale
from dataset.bbox import selective_search
import numpy as np

from models.ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork

# ハイパーパラメータ

# モデルのインスタンス化
network = ConvolutionalNeuralNetwork(input_dim=(1, 28, 28), 
                                   conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                                   hidden_size=100, output_size=10)

network.load_params("params.pkl")

with open('dataset/sample.bmp', 'rb') as f:
    file = File(f.read())

bmp = bmp.BMP()
bmp.read(file)

# bgr -> rgb
data = bmp.data[:, :, ::-1]
g_data = rgb2grayscale(data)

# Windowベース検出器
# 2 * 2から 画像の短い辺*短い辺までの正方形を全ピクセルにスライドさせる

# print g_data.shape
windows = selective_search(data)

predit = np.argmax(network.predict(windows))
if predit != 0:
    print predit, y, x