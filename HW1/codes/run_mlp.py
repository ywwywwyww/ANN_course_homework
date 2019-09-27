from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Linear
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
from sys import stderr

train_data, test_data, train_label, test_label = load_mnist_2d('data')

# Your model defintion here
# You should explore different model architecture

model = Network()
# model.add(Linear('fc1', 784, 10, 0.1, Sigmoid(name='sigmoid')))
model.add(Linear('fc1', 784, 200, 0.1, Relu(name='ReLU')))
model.add(Linear('fc2', 200, 100, 0.1, Relu(name='ReLU')))
model.add(Linear('fc3', 100, 50, 0.1, Relu(name='ReLU')))
model.add(Linear('fc4', 50, 20, 0.1, Relu(name='ReLU')))
model.add(Linear('fc5', 20, 10, 0.1, Relu(name='ReLU')))


# loss = SoftmaxCrossEntropyLoss(name='SoftmaxCrossEntropyLoss')
loss = EuclideanLoss(name='EuclideanLoss')
print('loss_function : %s' % (loss.name))

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 0.05,
    'weight_decay': 0.0,
    'momentum': 0.0,
    'training_batch_size': 100,
    'test_batch_size': 10000,
    'max_epoch': 1000,
    'disp_freq': 600,
    'test_epoch': 1
}

print(config)

for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    train_net(model, loss, config, train_data, train_label, config['training_batch_size'], config['disp_freq'])

    if epoch % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        test_net(model, loss, test_data, test_label, config['test_batch_size'])

    print('epoch %d finished , total = %d' % (epoch, config['max_epoch']), file=stderr)