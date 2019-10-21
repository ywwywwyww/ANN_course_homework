import os
import shutil

modelname = 'mlp_dropout_1000hiddennodes_droprate=0.1'
dir = 'log/' + modelname + '/'

class Initializer:
    def __init__(self):
        if not os.path.exists(dir):
            os.makedirs(dir)
        file1 = open(dir + 'train.txt', 'w')
        file2 = open(dir + 'val.txt', 'w')
        file3 = open(dir + 'test.txt', 'w')
        shutil.rmtree('train/')
initializer = Initializer()

def add_train(iters, loss, acc):
    file = open(dir + 'train.txt', 'a')
    print(iters, loss, acc, file=file)

def add_val(iters, loss, acc):
    file = open(dir + 'val.txt', 'a')
    print(iters, loss, acc, file=file)

def add_test(iters, loss, acc):
    file = open(dir + 'test.txt', 'a')
    print(iters, loss, acc, file=file)