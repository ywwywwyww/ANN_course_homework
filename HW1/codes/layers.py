import numpy as np
import math
import utils
import sys

class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        '''The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation'''

        self._saved_tensor = tensor


class Relu(Layer):
    def __init__(self, name='relu'):
        super(Relu, self).__init__(name)
        print('relu %s\n' % (name), file=utils.log_file)

    def f(self, input):
        return np.fmax(input, np.zeros(input.shape))

    def df(self, input):
        return (input >= 0).astype(np.float64)

    def forward(self, input):
        '''Your codes here'''
        self._saved_for_backward(input)
        return self.f(input)

    def backward(self, grad_output):
        '''Your codes here'''
        return np.multiply(grad_output, self.df(self._saved_tensor))


class Sigmoid(Layer):
    def __init__(self, name='sigmoid'):
        super(Sigmoid, self).__init__(name)
        print('sigmoid %s\n' % (name), file=utils.log_file)

    def f(self, input):
        # return 1/(1 + math.exp(-input))
        return 1/(1 + np.exp(-input))

    def df(self, input):
        fx = self.f(input)
        return np.multiply(fx, (1 - fx))

    def forward(self, input):
        '''Your codes here'''
        self._saved_for_backward(input)
        return self.f(input)

    def backward(self, grad_output):
        '''Your codes here'''
        return np.multiply(grad_output, self.df(self._saved_tensor))


class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std = 0, activation_function = ''):
        super(Linear, self).__init__(name, trainable=True)

        if activation_function == 'Relu' or activation_function == 'relu':
            init_std = math.sqrt(4 / (in_num + out_num))
        elif activation_function == 'Sigmoid' or activation_function == 'sigmoid':
            init_std = math.sqrt(2 / (in_num + out_num))

        print('layer %s : in_num = %d , out_num = %d , init_std = %.5f' %  (name, in_num, out_num, init_std), file=utils.log_file)

        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        # self.b = np.zeros(out_num)
        self.b = np.random.randn(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

        self.initial = 0


    def forward(self, input):
        '''Your codes here'''
        self._saved_for_backward(input)
        # print(input.shape, self.W.shape, self.b.shape)
        return np.dot(input, self.W) + self.b

    def backward(self, grad_output):
        '''Your codes here'''
        # print(self.W[0], file=sys.stderr)
        # print(grad_output[0][:6], file=sys.stderr)
        input = self._saved_tensor
        self.grad_W = np.dot(np.transpose(input), grad_output)
        self.grad_b = np.dot(np.ones((1, input.shape[0])), grad_output)[0]
        return np.dot(grad_output, np.transpose(self.W))

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        if self.initial:
            self.diff_W = self.grad_W + wd * self.W
            self.diff_b = self.grad_b + wd * self.b
            self.initial = 0
        else:
            self.diff_W = mm * self.diff_W + (1 - mm) * (self.grad_W + wd * self.W)
            self.diff_b = mm * self.diff_b + (1 - mm) * (self.grad_b + wd * self.b)

        self.W = self.W - lr * self.diff_W
        self.b = self.b - lr * self.diff_b


class LeakyRelu(Layer):
    def __init__(self, name='leakyrelu'):
        super(LeakyRelu, self).__init__(name)
        print('Leakyrelu %s\n' % (name), file=utils.log_file)

    def f(self, input):
        # return np.fmax(input, np.zeros(input.shape))
        return np.fmax(input, input * 0.01)

    def df(self, input):
        # return (input > 0).astype(np.float64)
        return (input > 0).astype(np.float64) + (input <= 0).astype(np.float64) * 0.01

    def forward(self, input):
        '''Your codes here'''
        self._saved_for_backward(input)
        return self.f(input)

    def backward(self, grad_output):
        '''Your codes here'''
        return np.multiply(grad_output, self.df(self._saved_tensor))

class Normalization(Layer):
    def __init__(self, name='normalization'):
        super(Normalization, self).__init__(name)
        print('normalization %s\n' % (name), file=utils.log_file)

    def forward(self, input):
        '''Your codes here'''
        mean = np.mean(input, axis=1)
        std = np.std(input, axis=1)
        return np.divide((input - mean.reshape(input.shape[0], 1)), std.reshape(input.shape[0], 1))

    def backward(self, grad_output):
        '''Your codes here'''
        return grad_output