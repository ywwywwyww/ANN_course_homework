from network import Network
from utils import LOG_INFO
import utils
from layers import Relu, Sigmoid, Linear, LeakyRelu
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss, CrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
from sys import stderr
from datetime import datetime

train_data, test_data, train_label, test_label = load_mnist_2d('data')

now = datetime.now()
utils.log_file = open('log\\%04d_%02d_%02d_%02d_%02d_%02d.txt' % (now.year, now.month, now.day, now.hour, now.minute, now.second), "w")

# Your model defintion here
# You should explore different model architecture
model = Network()
model.add(Linear('fc1', 784, 10, 0.05))
# model.add(Relu())
model.add(Sigmoid())
# model.add(LeakyRelu())

# loss = CrossEntropyLoss(name='CrossEntropyLoss')
# loss = SoftmaxCrossEntropyLoss(name='SoftmaxCrossEntropyLoss')
loss = EuclideanLoss(name='EuclideanLoss')
print('loss_function : %s' % (loss.name), file=utils.log_file)

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate_a' : 10,
    'learning_rate_b' : 20,
    'learning_rate': 0.01,
    'weight_decay': 0,
    'momentum': 0.0,
    'training_batch_size': 100,
    'test_batch_size': 10000,
    'max_epoch': 10000,
    'disp_freq': 600,
    'test_epoch': 1
}

print(config, file=utils.log_file)

for epoch in range(config['max_epoch']):

    # config['learning_rate'] = config['learning_rate_a'] / (config['learning_rate_b'] + epoch)

    LOG_INFO('Training @ %d epoch...' % (epoch))
    training_acc = train_net(model, loss, config, train_data, train_label, config['training_batch_size'], config['disp_freq'])
    test_acc = 0

    if epoch % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        test_acc = test_net(model, loss, test_data, test_label, config['test_batch_size'])

    print('epoch %d finished , total = %d , training acc = %.5f , test acc = %.5f' % (epoch, config['max_epoch'], training_acc, test_acc), file=stderr)

    utils.log_file.flush()