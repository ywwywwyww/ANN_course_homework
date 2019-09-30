import numpy as np


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
    def __init__(self, name):
        super(Relu, self).__init__(name)

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
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)

    def f(self, input):
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
    def __init__(self, name, in_num, out_num, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

    def forward(self, input):
        '''Your codes here'''
        self._saved_for_backward(input)
        return np.dot(input, self.W) + self.b

    def backward(self, grad_output):
        '''Your codes here'''
        input = self._saved_tensor
        self.grad_W = np.dot(np.transpose(input), grad_output)
        self.grad_b = np.dot(np.ones((1, input.shape[0])), grad_output)[0]
        return np.dot(grad_output, np.transpose(self.W))

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b
