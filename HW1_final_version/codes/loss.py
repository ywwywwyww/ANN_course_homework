from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here'''
        return np.sum(np.multiply(input - target, input - target)) / (2 * input.shape[0])

    def backward(self, input, target):
        '''Your codes here'''
        return (input - target) / input.shape[0]


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here'''
        exp_input = np.exp(input)
        exp_input_sum = np.dot(exp_input, np.ones((exp_input.shape[1], 1)))
        output = np.divide(exp_input, exp_input_sum)
        loss = np.sum(-(np.multiply(target, np.log(output)))) / input.shape[0]
        return loss

    def backward(self, input, target):
        '''Your codes here'''
        exp_input = np.exp(input)
        exp_input_sum = np.dot(exp_input, np.ones((exp_input.shape[1], 1)))
        output = np.divide(exp_input, exp_input_sum)
        return (output - target) / input.shape[0]
