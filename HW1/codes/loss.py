from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here'''
        return np.dot(np.multiply(input - target, input - target), np.ones((input.shape[1], 1))) / (2 * input.shape[1])

        pass

    def backward(self, input, target):
        '''Your codes here'''
        return input - target

        pass


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here'''
        exp_input = np.exp(input)
        exp_input_sum = np.dot(exp_input, np.ones((exp_input.shape[1],1)))
        output = np.divide(exp_input, exp_input_sum)
        loss = -np.multiply(target, np.log(output))
        return loss

        pass

    def backward(self, input, target):
        '''Your codes here'''
        return input - target

        pass
