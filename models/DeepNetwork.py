# coding: utf-8
import numpy as np
from collections import OrderedDict
import pickle
from models.layers import Convolution, Pooling, Relu, Affine, SoftmaxWithLoss, BatchNormalization, Dropout
from models.functions import sigmoid

class VGG16:
    # like VGG16 network
    
    def __init__(self, input_dim=(3, 224, 224), output_size=9, padding=0, stride=1):
        self.input_dim = input_dim
        self.output_size = output_size
        self.padding = padding
        self.stride = stride
        
        self.params = {}
        self.params['conv1_1_W'] = np.random.randn(64, input_dim[0], 3, 3) / np.sqrt(64 * 3 * 3)
        self.params['conv1_1_b'] = np.zeros(64)
        self.params['conv1_2_W'] = np.random.randn(64, 64, 3, 3) / np.sqrt(64 * 3 * 3)
        self.params['conv1_2_b'] = np.zeros(64)
        
        self.params['conv2_1_W'] = np.random.randn(128, 64, 3, 3) / np.sqrt(128 * 3 * 3)
        self.params['conv2_1_b'] = np.zeros(128)
        self.params['conv2_2_W'] = np.random.randn(128, 128, 3, 3) / np.sqrt(128 * 3 * 3)
        self.params['conv2_2_b'] = np.zeros(128)
        
        self.params['conv3_1_W'] = np.random.randn(256, 128, 3, 3) / np.sqrt(256 * 3 * 3)
        self.params['conv3_1_b'] = np.zeros(256)
        self.params['conv3_2_W'] = np.random.randn(256, 256, 3, 3) / np.sqrt(256 * 3 * 3)
        self.params['conv3_2_b'] = np.zeros(256)
        self.params['conv3_3_W'] = np.random.randn(256, 256, 3, 3) / np.sqrt(256 * 3 * 3)
        self.params['conv3_3_b'] = np.zeros(256)
        
        self.params['conv4_1_W'] = np.random.randn(512, 256, 3, 3) / np.sqrt(512 * 3 * 3)
        self.params['conv4_1_b'] = np.zeros(512)
        self.params['conv4_2_W'] = np.random.randn(512, 512, 3, 3) / np.sqrt(512 * 3 * 3)
        self.params['conv4_2_b'] = np.zeros(512)
        self.params['conv4_3_W'] = np.random.randn(512, 512, 3, 3) / np.sqrt(512 * 3 * 3)
        self.params['conv4_3_b'] = np.zeros(512)
        
        self.params['conv5_1_W'] = np.random.randn(512, 512, 3, 3) / np.sqrt(512 * 3 * 3)
        self.params['conv5_1_b'] = np.zeros(512)
        self.params['conv5_2_W'] = np.random.randn(512, 512, 3, 3) / np.sqrt(512 * 3 * 3)
        self.params['conv5_2_b'] = np.zeros(512)
        self.params['conv5_3_W'] = np.random.randn(512, 512, 3, 3) / np.sqrt(512 * 3 * 3)
        self.params['conv5_3_b'] = np.zeros(512)
        
        
        self.params['fc6_W'] = np.random.randn(41472, 4096) / np.sqrt(41472)
        self.params['fc6_b'] = np.zeros(4096)
        self.params['fc7_W'] = np.random.randn(4096, 4096) / np.sqrt(4096)
        self.params['fc7_b'] = np.zeros(4096)
        
        self.params['cls_W'] = np.random.randn(4096, output_size * 2) / np.sqrt(4096)
        self.params['cls_b'] = np.zeros(output_size * 2)
        self.params['regr_W'] = np.random.randn(output_size * 2, output_size) / np.sqrt(4096)
        self.params['regr_b'] = np.zeros(output_size)
        
        self.layers = OrderedDict()
        self.layers['conv1_1'] = Convolution(self.params['conv1_1_W'], self.params['conv1_1_b'], self.stride, self.padding)
        self.layers['relu1_1'] = Relu()
        self.layers['conv1_2'] = Convolution(self.params['conv1_2_W'], self.params['conv1_2_b'], self.stride, self.padding)
        self.layers['relu1_2'] = Relu()
        self.layers['pool1'] = Pooling(2, 2, 2)
        
        self.layers['conv2_1'] = Convolution(self.params['conv2_1_W'], self.params['conv2_1_b'], self.stride, self.padding)
        self.layers['relu2_1'] = Relu()
        self.layers['conv2_2'] = Convolution(self.params['conv2_2_W'], self.params['conv2_2_b'], self.stride, self.padding)
        self.layers['relu2_2'] = Relu()
        self.layers['pool2'] = Pooling(2, 2, 2)
        
        self.layers['conv3_1'] = Convolution(self.params['conv3_1_W'], self.params['conv3_1_b'], self.stride, self.padding)
        self.layers['relu3_1'] = Relu()
        self.layers['conv3_2'] = Convolution(self.params['conv3_2_W'], self.params['conv3_2_b'], self.stride, self.padding)
        self.layers['relu3_2'] = Relu()
        self.layers['conv3_3'] = Convolution(self.params['conv3_3_W'], self.params['conv3_3_b'], self.stride, self.padding)
        self.layers['relu3_3'] = Relu()
        self.layers['pool3'] = Pooling(2, 2, 2)
        self.layers['conv4_1'] = Convolution(self.params['conv4_1_W'], self.params['conv4_1_b'], self.stride, self.padding)
        self.layers['relu4_1'] = Relu()
        self.layers['conv4_2'] = Convolution(self.params['conv4_2_W'], self.params['conv4_2_b'], self.stride, self.padding)
        self.layers['relu4_2'] = Relu()
        self.layers['conv4_3'] = Convolution(self.params['conv4_3_W'], self.params['conv4_3_b'], self.stride, self.padding)
        self.layers['relu4_3'] = Relu()
        self.layers['pool4'] = Pooling(2, 2, 2)
        self.layers['conv5_1'] = Convolution(self.params['conv5_1_W'], self.params['conv5_1_b'], self.stride, self.padding)
        self.layers['relu5_1'] = Relu()
        self.layers['conv5_2'] = Convolution(self.params['conv5_2_W'], self.params['conv5_2_b'], self.stride, self.padding)
        self.layers['relu5_2'] = Relu()
        self.layers['conv5_3'] = Convolution(self.params['conv5_3_W'], self.params['conv5_3_b'], self.stride, self.padding)
        self.layers['relu5_3'] = Relu()
        self.layers['pool5'] = Pooling(2, 2, 2)
        self.layers['fc6'] = Affine(self.params['fc6_W'], self.params['fc6_b'])
        self.layers['relu6'] = Relu()
        self.layers['fc7'] = Affine(self.params['fc7_W'], self.params['fc7_b'])
        self.layers['relu7'] = Relu()
        self.layers['cls'] = Affine(self.params['cls_W'], self.params['cls_b'])
        self.layers['regr'] = Affine(self.params['regr_W'], self.params['regr_b'])
        self.last_layer = SoftmaxWithLoss()
    
    def save_params(self, file_name="rpn_params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
    
    def load_params(self, file_name="rpn_params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
        
        for layer_name in self.layers.keys():
            self.layers[layer_name].W = self.params[layer_name + '_W']
            self.layers[layer_name].b = self.params[layer_name + '_b']
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)
    
    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        
        # backward
        dout = 1
        dout = self.last_layer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        for layer_name in self.layers.keys():
            if hasattr(self.layers[layer_name], 'dW'):
                grads[layer_name + '_W'] = self.layers[layer_name].dW
                grads[layer_name + '_b'] = self.layers[layer_name].db
        
        return grads
    
    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]
