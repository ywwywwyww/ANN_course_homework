from network import Network
from utils import LOG_INFO
import utils
from layers import Relu, Sigmoid, Linear, LeakyRelu, Normalization, Tanh
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss, CrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
from sys import stderr
from datetime import datetime
import draw
import math

train_data, test_data, train_label, test_label = load_mnist_2d('data')

now = datetime.now()
utils.log_file = open('log\\%04d_%02d_%02d_%02d_%02d_%02d.txt' % (now.year, now.month, now.day, now.hour, now.minute, now.second), "w")
draw.plotfilename = 'log\\%04d_%02d_%02d_%02d_%02d_%02d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)

# Your model defintion here
# You should explore different model architecture
model = Network()
model.add(Normalization())
model.add(Linear('fc1', 784, 100, activation_function='Tanh'))
model.add(Tanh())
model.add(Linear('fc2', 100, 10, activation_function='Tanh'))
model.add(Tanh())

# loss = CrossEntropyLoss(name='CrossEntropyLoss')
loss = SoftmaxCrossEntropyLoss(name='SoftmaxCrossEntropyLoss')
# loss = EuclideanLoss(name='EuclideanLoss')
print('loss_function : %s' % (loss.name), file=utils.log_file)

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'training_set_size' : 60000,
    'test_set_size' : 10000,
    'learning_rate_a' : 10,
    'learning_rate_b' : 20,
    'learning_rate': 0.01,
    'weight_decay': 0.0,
    'momentum': 0.95,
    'batch_size': 100,
    'max_epoch': 1000,
    'disp_freq': 600,
    'test_epoch': 1
}

print(config, file=utils.log_file)

for epoch in range(config['max_epoch']):

    # config['learning_rate'] = config['learning_rate_a'] / (config['learning_rate_b'] + epoch)

    LOG_INFO('Training @ %d epoch...' % (epoch))
    training_acc = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
    test_loss, test_acc, train_acc, train_loss = 0, 0, 0, 0

    if epoch % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        test_loss, test_acc = test_net(model, loss, test_data, test_label, config['test_set_size'])
        train_loss, train_acc = test_net(model, loss, train_data, train_label, config['training_set_size'])

    print('epoch %d finished , total = %d , training loss = %.5f , training acc = %.5f , test loss = %.5f , test acc = %.5f' % (epoch, config['max_epoch'], train_loss, train_acc, test_loss, test_acc), file=stderr)

    draw.plot.add_test((epoch + 1) * (config['training_set_size'] / config['batch_size']), test_loss, test_acc)
    draw.plot.add_training((epoch + 1) * (config['training_set_size'] / config['batch_size']), train_loss, train_acc)

    draw.plot.draw()

    utils.log_file.flush()