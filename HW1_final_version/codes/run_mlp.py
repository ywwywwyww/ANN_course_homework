from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Linear
from loss import EuclideanLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
from math import sqrt
from draw import plot


train_data, test_data, train_label, test_label = load_mnist_2d('data')

# Your model defintion here
# You should explore different model architecture
model = Network()
model.add(Linear('fc1', 784, 100, sqrt(2 / (784 + 100))))
model.add(Sigmoid(name="Sigmoid"))
model.add(Linear('fc2', 100, 10, sqrt(2 / (100 + 10))))
model.add(Sigmoid(name="Sigmoid"))

loss = EuclideanLoss(name='loss')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 0.01,
    'weight_decay': 0.0,
    'momentum': 0.0,
    'batch_size': 100,
    'max_epoch': 1000,
    'disp_freq': 600,
    'test_epoch': 1
}

for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])

    if epoch % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        training_loss, training_acc = test_net(model, loss, train_data, train_label, train_data.shape[0])
        test_loss, test_acc = test_net(model, loss, test_data, test_label, test_data.shape[0])
        plot.add_training((epoch + 1) * train_data.shape[0] // config['batch_size'], training_loss, training_acc)
        plot.add_test((epoch + 1) * train_data.shape[0] // config['batch_size'], test_loss, test_acc)

plot.draw()