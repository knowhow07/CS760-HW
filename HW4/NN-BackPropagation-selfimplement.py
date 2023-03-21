#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 22:14:38 2023

@author: nuohaoliu
some ideas are inspired by Omar Aflak
"""
# from keras.datasets import mnist
from mnist import MNIST
import numpy as np
# import torch

# import matplotlib.pyplot as plt
# import math
# from scipy.special import expit
import sys
import time
from datetime import timedelta
start_time = time.monotonic()

# image_size = 28 # width and length
# no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
# image_pixels = image_size * image_size
# data_path = "data/"
# train_data = np.loadtxt(data_path + "mnist_train.csv", 
#                         delimiter=",")
# test_data = np.loadtxt(data_path + "mnist_test.csv", 
#                        delimiter=",") 
# test_data[:10]
# (train_X, train_y), (test_X, test_y) = mnist.load_data()

# print ((train_X, train_y))
image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
data_path = "data/"
train_data = np.loadtxt(data_path + "mnist_train.csv", 
                        delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv", 
                        delimiter=",") 
#%%

        
# def log_loss(y, y_pred):
#     log_loss = -np.mean(y * np.log(y_pred) + (1-y) * np.log(1 - y_pred))
#     return log_loss


import numpy as np

# Basic Layer
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X
    def forward_propagation(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
    
# Fully Connected Layer
class layer_fullyconnected(Layer):

    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
    
# inherit from basic layer
class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error
    
    import numpy as np



def sigmoid(x):
    return 1./(1. + np.exp(-x))

def sigmoid_d(x):
    return sigmoid(x)*(1.-sigmoid(x))

def softmax(x):
    m = np.max(x) #prevent overflow
    # beta = torch.max(x, dim=1, keepdim=True)[0] # avoid overflow 
    return(np.exp(x-m)/np.exp(x-m).sum())

# if i == j:
#     self.gradient[i,j] = self.value[i] * (1-self.value[i])
# else: 
#     self.gradient[i,j] = -self.value[i] * self.value[j]
    
def softmax_d(x):

    # beta = torch.max(x, dim=1, keepdim=True)[0] # avoid overflow 
    jacobian = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            if i == j:
                jacobian[i, j] = np.dot(np.reshape(softmax(x)[i],(1,len(x[i]))) 
                , np.reshape((1 - softmax(x)[i]),(len(x[i]),1)))
            else:
                jacobian[i, j] = np.dot(np.reshape(softmax(x)[i],(1,len(x[i]))), 
                                                    np.reshape(softmax(x)[j],(len(x[j]),1)))

    return jacobian

    return jacobian

# def sigmoid_d(x):
#     return 1 / (1 + np.exp(-x))*(1-1 / (1 + np.exp(-x)))



# loss function and its derivative
def cross_entropy (y_true, y_pred):
    return np.mean(-y_true * np.log10(y_pred));

def cross_entropy_d (y_true, y_pred):
    return np.sum(-y_true/y_pred);

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));

def mse_d(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result
    
    def error(self,out,y):
        correct = 0
        out =np.array(np.reshape(out,(len(out),10)))
        # out5 =np.array(np.reshape(out[0:5],(5,10)))
        # out5 = np.where(out5 > 0.5, 1, 0)
        out = np.where(out > 0.5, 1, 0)
        for i in range(0,len(out)):
            if np.array_equal(out[i],y[i]):
                correct +=1
        test_loss = np.absolute(y-out)
        ave_loss = np.sum(test_loss)/len(y)
        error = (len(y)-correct)/len(y) 
        return error,correct
    
    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        assert inputs.shape[0] == targets.shape[0]
        if shuffle:
            indices = np.arange(inputs.shape[0])
            np.random.shuffle(indices)
        for start_idx in range(0, inputs.shape[0], batchsize):
            end_idx = min(start_idx + batchsize, inputs.shape[0])
            if shuffle:
                excerpt = indices[start_idx:end_idx]
            else:
                excerpt = slice(start_idx, end_idx)
            yield inputs[excerpt], targets[excerpt]
    
    # train the network
    def fit(self, x_train, y_train, x_test, y_test, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)
        learning_curve = np.empty((0,3))
        errp_min = 1.0
        # training loop
        for i in range(epochs):
            err = 0
            # for batch in self.iterate_minibatches(x_train, y_train, 128, shuffle=False):
            #     x_batch, y_batch = batch
            #     samples = len(x_batch)
            
            #     for j in range(samples):
            #         # forward propagation
            #         output = x_batch[j]
            #         print
            #         for layer in self.layers:
            #             output = layer.forward_propagation(output)
    
            #         # compute loss (for display purpose only)
            #         err += self.loss(y_batch[j], output)
    
            #         # backward propagation
            #         error = self.loss_prime(y_batch[j], output)
            #         for layer in reversed(self.layers):
            #             error = layer.backward_propagation(error, learning_rate)
            for j in range(samples):
                output = x_train[j]
                print
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                err += self.loss(y_train[j], output)

                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
            err /= samples
            
            out_train = self.predict(x_train)
            errp,count = self.error(out_train,y_train)
            # if errp < errp_min:
            #     errp_min = errp
            #     np.save('model_best.npy', output)
            #     torch.save(model_self, 'model_self_best.pth.tar')
            #     out_test = self.predict(x_test)
            #     errp_test,count_test = self.error(out_test,y_test)             
            #     print('epoch %d/%d   tr_err=%.4f  te_no=%d  tr_err=%.4f  te_no=%d'
            #           % (i+1, epochs, errp, count, errp_test, count_test))
            #     learning_curve = np.append(learning_curve,np.reshape([i+1,errp,errp_test],(1,3)),axis=0)
            # else: 
            #     i = i - 1
            #     for layer in self.layers:
            #         output = layer.forward_propagation(np.load('model_best.npy'))
            
            out_test = self.predict(x_test)
            errp_test,count_test = self.error(out_test,y_test)             
            print('epoch %d/%d   tr_err=%.4f  te_no=%d  tr_err=%.4f  te_no=%d'
                  % (i+1, epochs, errp, count, errp_test, count_test))
            learning_curve = np.append(learning_curve,np.reshape([i+1,errp,errp_test],(1,3)),axis=0)
       
        return learning_curve
            
    # training data
# x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
# y_train = np.array([[[0]], [[1]], [[1]], [[0]]])



a, b = np.shape(test_data)
x_train = np.array(train_data[:,1:b])
y_train = np.array(train_data[:,0],int)
x_test = np.array(test_data[:,1:b])
y_test = np.array(test_data[:,0],int)
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
# x_train /= 255
# # encode output which is a number in range [0,9] into a vector of size 10
# # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
# y_train = np_utils.to_categorical(y_train)
res = np.zeros((y_train.size, 10), dtype=int)
res[np.arange(y_train.size), y_train] = 1
y_train = res

# # same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255


res = np.zeros((y_test.size, 10), dtype=int)
res[np.arange(y_test.size), y_test] = 1
y_test = res
# np.eye(10)[y_test]
# y_test = np_utils.to_categorical(y_test)
# network

# net = Network()
# net.add(layer_fullyconnected(2, 3))
# # net.add(ActivationLayer(tanh, tanh_prime))
# net.add(ActivationLayer(sigmoid, sigmoid_d))
# net.add(layer_fullyconnected(3, 1))
# net.add(ActivationLayer(sigmoid, sigmoid_d))

# # train
# net.use(cross_entropy, cross_entropy_d)
# # net.use(mse, mse_d)
# lcurve = net.fit(x_train, y_train, epochs=100, learning_rate=0.1)

# # test
# out = net.predict(x_train)
# print(out)

net = Network()
net.add(layer_fullyconnected(28*28, 300))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net.add(ActivationLayer(sigmoid, sigmoid_d))
net.add(layer_fullyconnected(300, 200))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(ActivationLayer(sigmoid, sigmoid_d))
net.add(layer_fullyconnected(200, 10))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
net.add(ActivationLayer(sigmoid, sigmoid_d))
# net.add(ActivationLayer(softmax, softmax_d))

# train on 1000 samples
# as we didn't implemented mini-batch GD, training will be pretty 
#slow if we update at each iteration on 60000 samples...
net.use(mse, mse_d)
# net.use(cross_entropy,cross_entropy_d)
# lcurve = net.fit(x_train[1:1000], y_train[1:1000], 
#                   x_test[1:100], y_test[1:100],epochs=10, learning_rate=0.05)
lcurve = net.fit(x_train, y_train, 
                  x_test, y_test,epochs=30, learning_rate=0.1)
# lcurve = net.fit(x_train, y_train, epochs=1, learning_rate=0.1)

# test on 3 samples
out = net.predict(x_test)
#%%
import numpy as np
out =np.array(np.reshape(out,(10000,10)))
# out5 =np.array(np.reshape(out[0:5],(5,10)))
# out5 = np.where(out5 > 0.5, 1, 0)
out = np.where(out > 0.5, 1, 0)
correct = 0

def error(out,y):
    correct = 0
    for i in range(0,len(out)):
        if np.array_equal(out[i],y[i]):
            correct +=1
    test_loss = np.absolute(y-out)
    ave_loss = np.sum(test_loss)/len(y)
    error = (len(y)-correct)/len(y) 
    print(len(y),len(out))
    return error,correct,ave_loss

end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time)) 
# error,correct,ave_loss = error(out,y_test)
# print(error,correct)
# test_loss = np.absolute(y_test[0:5]-out[0:5])
# test_lossssum = np.sum(test_loss)
# y10 = y_test[0:5]
# print("\n")
# print("predicted values : ")
# print(out, end="\n")
# print("true values : ")
# print(y_test)