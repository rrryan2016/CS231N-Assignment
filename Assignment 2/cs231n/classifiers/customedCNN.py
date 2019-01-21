# Made by ghostInSh3ll <liuwz2017@gmail.com>
# -*- coding:utf-8 -*-

# As usual, a bit of setup

import numpy as np
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class CustomedConvNet(object):
    '''
    the structure is: INPUT --> [CONV --> RELU --> POOL]*2 --> [CONV --> RELU] --> FC/OUT
    in cnn.py the structure is: conv - relu - 2x2 max pool - affine - relu - affine - softmax 
    '''
    def __init__(self, input_dim=(3,32,32), num_filters=32, filter_size = 5,
                 hidden_dim = 100, num_classes =10, weight_scale = 1e-3, reg = 0.0 ,
                 dtype=np.float32):
        # Initialization 
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        C, H, W = input_dim # Determined by CIFAR-10
        F, HH, WW = num_filters, filter_size, filter_size
        ## TODO:

        # self.params['W1'] = weight_scale * np.random.randn(F,C,filter_size,filter_size)
        # self.params['W2'] = weight_scale * np.random.randn(F,F,filter_size,filter_size)
        # # self.params['W3'] = weight_scale * np.random.randn(16384, 50)
        # self.params['W3'] = weight_scale * np.random.randn(F*H//2*W//2,hidden_dim*2)
        # # self.params['W3'] = weight_scale * np.random.randn(F*H*W,hidden_dim)
        # self.params['W4'] = weight_scale * np.random.randn(hidden_dim*2,hidden_dim)
        # self.params['W5'] = weight_scale * np.random.randn(hidden_dim,num_classes)
        #
        # self.params['b1'] = np.zeros(F)
        # self.params['b2'] = np.zeros(F)
        # self.params['b3'] = np.zeros(hidden_dim*2)
        # self.params['b4'] = np.zeros(hidden_dim)
        # self.params['b5'] = np.zeros(num_classes)

        self.params['W1'] = weight_scale * np.random.randn(F,C,HH,WW)
        self.params['W2'] = weight_scale * np.random.randn(F,F,HH,WW)
        self.params['W3'] = weight_scale * np.random.randn(F,F,HH,WW)
        self.params['W4'] = weight_scale * np.random.randn(F*H//4*W//4,num_classes)
        # self.params['W5'] = weight_scale * np.random.randn(hidden_dim,num_classes)

        self.params['b1'] = np.zeros(F)
        self.params['b2'] = np.zeros(F)
        self.params['b3'] = np.zeros(F)
        self.params['b4'] = np.zeros(num_classes)
        # self.params['b5'] = np.zeros(num_classes)

        print(self.params['W1'].shape)
        print(self.params['W2'].shape)
        print(self.params['W3'].shape)
        print(self.params['W4'].shape)
        # print(self.params['W5'].shape)

        print(self.params['b1'].shape)
        print(self.params['b2'].shape)
        print(self.params['b3'].shape)
        print(self.params['b4'].shape)
        # print(self.params['b5'].shape)


        for k,v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y = None):
        W1, b1 = self.params['W1'],self.params['b1']
        W2, b2 = self.params['W2'],self.params['b2']
        W3, b3 = self.params['W3'],self.params['b3']
        W4, b4 = self.params['W4'],self.params['b4']
        # W5, b5 = self.params['W5'],self.params['b5']

        filter_size = W1.shape[2]
        conv_param = {'stride':1,'pad':(filter_size-1)//2}
        pool_param = {'pool_height':2, 'pool_width':2, 'stride':2}

        scores = None 
        # conv_out_1 , conv_cache_1 = conv_relu_forward(X,W1,b1,conv_param)
        # conv_out_2 , conv_cache_2 = conv_relu_forward(conv_out_1,W2,b2,conv_param)
        # affine_out_1, affine_cache_1 = affine_relu_forward(conv_out_2,W3,b3)

        # conv_out , conv_cache = conv_relu_forward(X,W1,b1,conv_param)
        # pool_out , pool_cache = conv_relu_pool_forward(conv_out,W2,b2,conv_param,pool_param)
        # affine_out_1, affine_cache_1 = affine_relu_forward(pool_out,W3,b3)
        # affine_out_2, affine_cache_2 = affine_relu_forward(affine_out_1,W4,b4)
        # scores, cache = affine_forward(affine_out_2,W5,b5)

        pool_out_1, pool_cache_1 = conv_relu_pool_forward(X,W1,b1,conv_param,pool_param)
        pool_out_2, pool_cache_2 = conv_relu_pool_forward(pool_out_1,W2,b2,conv_param,pool_param)
        conv_out, conv_cache = conv_relu_forward(pool_out_2,W3,b3,conv_param)
        # affine_out, affine_cache = affine_relu_forward(conv_out,W4,b4)
        # affine_out, affine_cache = affine_relu_forward(pool_out,W2,b2)
        scores, cache = affine_forward(conv_out,W4,b4)

        if y is None:
            return scores 
        loss, grads = 0, {}

        # Backward
        loss, dscore = softmax_loss(scores, y)
        daffine, grads['W4'], grads['b4'] = affine_backward(dscore,cache)
        dconv, grads['W3'], grads['b3'] = conv_relu_backward(daffine, conv_cache)
        dpool, grads['W2'], grads['b2'] = conv_relu_pool_backward(dconv,pool_cache_2)
        dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dpool,pool_cache_1)

        # daffine_1, grads['W3'], grads['b3'] = 
        # daffine_2, grads['W2'], grads['b2'] = affine_relu_backward(daffine_1,affine_cache)
        # dx,grads['W1'], grads['b1'] = conv_relu_pool_backward(daffine_2,pool_cache)

        
        loss += 0.5 * self.reg * (np.sum(W1 ** 2)+np.sum(W2 ** 2)+np.sum(W3 ** 2)+np.sum(W4 ** 2))
        # loss += 0.5 * self.reg * (np.sum(W1 ** 2)+np.sum(W2 ** 2)+np.sum(W3 ** 2)+np.sum(W4 ** 2)+np.sum(W5 ** 2))

        grads['W1'] += self.reg * W1
        grads['W2'] += self.reg * W2
        grads['W3'] += self.reg * W3
        grads['W4'] += self.reg * W4
        # grads['W5'] += self.reg * W5

        return loss, grads 

pass

