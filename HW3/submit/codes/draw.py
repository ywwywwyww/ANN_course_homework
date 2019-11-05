import matplotlib.pyplot as plt
import numpy as np
import os

# Log = open("log\\log.txt", "w")

def ReadData(FileName):
    Iters = []
    Loss = []
    Acc = []
    File = open(FileName, "r")
    Data = File.readlines()
    for Line in Data:
        ItersStr, LossStr, AccStr = Line.split()
        # print(ItersStr, LossStr, AccStr)
        Iters.append(int(ItersStr))
        Loss.append(float(LossStr))
        Acc.append(float(AccStr))
    # print(Iters[:4])
    # print(Loss[:4])
    # print(Acc[:4])
    return Iters, Loss, Acc

def plot(FolderName):
    # FolderName = 'log\\one_hidden_layer_relu_crossentropy_lr=0.1_m=0_10000epochs_3'
    print(FolderName)
    # return
    Folder = FolderName + '\\'
    TestName = Folder + 'test.txt'
    TrainingName = Folder + 'training.txt'
    TestPerEpochName = Folder + 'perepoch_test.txt'
    TrainingPerEpochName = Folder + 'perepoch_training.txt'
    OutputImageName = FolderName + '.png'

    TrainingIters, TrainingLoss, TrainingAcc = ReadData(TrainingName)
    TestPerEpochIters, TestPerEpochLoss, TestPerEpochAcc = ReadData(TestPerEpochName)
    TrainingPerEpochIters, TrainingPerEpochLoss, TrainingPerEpochAcc = ReadData(TrainingPerEpochName)
    #
    # s1 = 1e100
    # s2 = 1e100
    # for i in range(0, TestPerEpochIters.__len__()):
    #     if TestPerEpochAcc[i] >= 0.95:
    #         s1 = min(s1, TestPerEpochIters[i])
    #     if TestPerEpochAcc[i] >= 0.98:
    #         s2 = min(s2, TestPerEpochIters[i])
    #
    # print(s1, s2, FolderName, file=Log)
    #
    # return

    dpi = 100
    fig, subplots = plt.subplots(2, 2, figsize=(1300 / dpi, 500 / dpi), dpi=dpi)
    # fig.figure(figsize=(3.841, 7.195), dpi=100)

    plt.subplots_adjust(hspace=0.6, left=0.1, right=0.95)

    subplots[0,0].plot(TrainingIters, TrainingLoss, color='blue', label='Training Set')
    subplots[0,0].set_title('Batch Loss During Training Per Iteration')
    subplots[0,0].set_xlabel('Iterations')
    subplots[0,0].set_ylabel('Loss')
    subplots[0,0].legend(loc='best')
    subplots[0,0].set_ylim(-0.01, 0.3)

    subplots[0,1].plot(TrainingIters, TrainingAcc, color='blue', label='Training Set')
    subplots[0,1].set_title('Batch Accuracy During Training Per Iteration')
    subplots[0,1].set_xlabel('Iterations')
    subplots[0,1].set_ylabel('Accuracy')
    subplots[0,1].legend(loc='best')
    subplots[0,1].set_ylim(0.9, 1.01)

    subplots[1,0].plot(TrainingPerEpochIters, TrainingPerEpochLoss, color='blue', label='Training Set')
    subplots[1,0].plot(TestPerEpochIters, TestPerEpochLoss, color='orange', label='Test Set')
    subplots[1,0].set_title('Loss During Training Per Epoch')
    subplots[1,0].set_xlabel('Iterations')
    subplots[1,0].set_ylabel('Loss')
    subplots[1,0].legend(loc='best')
    subplots[1,0].set_ylim(-0.01, 0.3)
    # MinLoss = min(TestPerEpochLoss)
    # MinIters = TestPerEpochIters[TestPerEpochLoss.index(MinLoss)]
    # print(MinIters, MinLoss)
    # if MinIters / TestPerEpochIters[-1] < 0.2:
    #     subplots[1,0].annotate(r'$(%d, %.4f)$' % (MinIters, MinLoss), xy=(MinIters, MinLoss), xycoords='data', xytext=(+50, +30),
    #                  textcoords='offset points', fontsize=16,
    #                  arrowprops=dict(arrowstyle='->'))
    # elif MinIters / TestPerEpochIters[-1] < 0.5:
    #     subplots[1,0].annotate(r'$(%d, %.4f)$' % (MinIters, MinLoss), xy=(MinIters, MinLoss), xycoords='data', xytext=(-30, +30),
    #                  textcoords='offset points', fontsize=16,
    #                  arrowprops=dict(arrowstyle='->'))
    # elif MinIters / TestPerEpochIters[-1] < 0.8:
    #     subplots[1,0].annotate(r'$(%d, %.4f)$' % (MinIters, MinLoss), xy=(MinIters, MinLoss), xycoords='data', xytext=(-100, +30),
    #                  textcoords='offset points', fontsize=16,
    #                  arrowprops=dict(arrowstyle='->'))
    # else:
    #     subplots[1,0].annotate(r'$(%d, %.4f)$' % (MinIters, MinLoss), xy=(MinIters, MinLoss), xycoords='data', xytext=(-200, +30),
    #                  textcoords='offset points', fontsize=16,
    #                  arrowprops=dict(arrowstyle='->'))

    subplots[1,1].plot(TrainingPerEpochIters, TrainingPerEpochAcc, color='blue', label='Training Set')
    subplots[1,1].plot(TestPerEpochIters, TestPerEpochAcc, color='orange', label='Test Set')
    subplots[1,1].set_title('Accuracy During Training Per Epoch')
    subplots[1,1].set_xlabel('Iterations')
    subplots[1,1].set_ylabel('Accuracy')
    subplots[1,1].legend(loc='best')
    subplots[1,1].set_ylim(0.9, 1.01)
    # MaxAcc = max(TestPerEpochAcc)
    # MaxIters = TestPerEpochIters[TestPerEpochAcc.index(MaxAcc)]
    # print(MaxIters, MaxAcc)
    # if MaxIters / TestPerEpochIters[-1] < 0.2:
    #     subplots[1,1].annotate(r'$(%d, %.4f)$' % (MaxIters, MaxAcc), xy=(MaxIters, MaxAcc), xycoords='data', xytext=(+50, -30),
    #                  textcoords='offset points', fontsize=16,
    #                  arrowprops=dict(arrowstyle='->'))
    # elif MaxIters / TestPerEpochIters[-1] < 0.5:
    #     subplots[1,1].annotate(r'$(%d, %.4f)$' % (MaxIters, MaxAcc), xy=(MaxIters, MaxAcc), xycoords='data', xytext=(-30, -30),
    #                  textcoords='offset points', fontsize=16,
    #                  arrowprops=dict(arrowstyle='->'))
    # elif MaxIters / TestPerEpochIters[-1] < 0.8:
    #     subplots[1,1].annotate(r'$(%d, %.4f)$' % (MaxIters, MaxAcc), xy=(MaxIters, MaxAcc), xycoords='data', xytext=(-100, -30),
    #                  textcoords='offset points', fontsize=16,
    #                  arrowprops=dict(arrowstyle='->'))
    # else:
    #     subplots[1,1].annotate(r'$(%d, %.4f)$' % (MaxIters, MaxAcc), xy=(MaxIters, MaxAcc), xycoords='data', xytext=(-200, -30),
    #                  textcoords='offset points', fontsize=16,
    #                  arrowprops=dict(arrowstyle='->'))

    plt.savefig(OutputImageName)
    # plt.show()

def plot2(FolderName1, FolderName2):
    print(FolderName1 + ' ' + FolderName2)
    Folder1 = FolderName1 + '\\'
    Folder2 = FolderName2 + '\\'
    TestPerEpochName1 = Folder1 + 'perepoch_test.txt'
    TrainingPerEpochName1 = Folder1 + 'perepoch_training.txt'
    TestPerEpochName2 = Folder2 + 'perepoch_test.txt'
    TrainingPerEpochName2 = Folder2 + 'perepoch_training.txt'
    OutputImageName = FolderName2 + '_2.png'

    TestPerEpochIters1, TestPerEpochLoss1, TestPerEpochAcc1 = ReadData(TestPerEpochName1)
    TestPerEpochIters2, TestPerEpochLoss2, TestPerEpochAcc2 = ReadData(TestPerEpochName2)


    dpi = 100
    fig, subplots = plt.subplots(2, 2, figsize=(1300 / dpi, 500 / dpi), dpi=dpi)
    # fig.figure(figsize=(3.841, 7.195), dpi=100)

    plt.subplots_adjust(hspace=0.6, left=0.1, right=0.95)

    subplots[0].plot(TestPerEpochIters1, TestPerEpochLoss1, color='blue', label='Without Normalization')
    subplots[0].plot(TestPerEpochIters2, TestPerEpochLoss2, color='orange', label='With Normalization')
    subplots[0].set_title('Loss During Training Per Epoch')
    subplots[0].set_xlabel('Iterations')
    subplots[0].set_ylabel('Loss')
    subplots[0].legend(loc='best')
    subplots[0].set_ylim(-0.01, 0.3)

    subplots[1].plot(TestPerEpochIters1, TestPerEpochAcc1, color='blue', label='Without Normalization')
    subplots[1].plot(TestPerEpochIters2, TestPerEpochAcc2, color='orange', label='With Normalization')
    subplots[1].set_title('Accuracy During Training Per Epoch')
    subplots[1].set_xlabel('Iterations')
    subplots[1].set_ylabel('Accuracy')
    subplots[1].legend(loc='best')
    subplots[1].set_ylim(0.9, 1.01)

    plt.savefig(OutputImageName)
#
# OutputFile = open("log\\b.txt", "w")
#
# for root, dirs, files in os.walk('log\\'):
#     if root == 'log\\':
# #         for file in files:
# #             if file[-5] != '2':
# #                 continue
# #             if file[-3:] == 'png':
# #                 # print('codes\\log\\' + file)
# #                 strs = file.split('_')
# #                 layer, act, loss, nor = "", "", "", ""
# #                 if strs[0] == 'one':
# #                     layer = 'one hidden layer'
# #                 else:
# #                     layer = 'two hidden layer'
# #                 if strs[3] == 'relu':
# #                     act = 'ReLU'
# #                 else:
# #                     act = 'Sigmoid'
# #                 if strs[4] == 'crossentropy':
# #                     loss = 'Softmax Cross-Entropy Loss'
# #                 else:
# #                     loss = 'Mean Square Error'
# #                 if file[-7] == 'n':
# #                     nor = 'With Normalization'
# #                 else:
# #                     nor = 'Without Normalization'
# # #                 print("<center>\n<img src=\"%s\">\n\
# # # <br>\n\
# # # <div>%s, %s, %s, %s</div>\n\
# # # </center>\n\n" % ('codes\\log\\' + file, layer, act, loss, nor), file=OutputFile)
# #                 print("<center>\n<img src=\"%s\">\n\
# # <br>\n\
# # <div>%s, %s, %s</div>\n\
# # </center>\n\n" % ('codes\\log\\' + file, layer, act, loss), file=OutputFile)
#
#         for dir in dirs:
#             if dir[-1] == '3':
#                 plot(root + dir)
#                 # for dir2 in dirs:
#                 #     if dir2[-1] == '3' and dir2.__len__() > dir.__len__() and dir2[:40] == dir[:40] :
#                 #         # print(dir,' ', dir2)
#                 #         plot2(root + dir, root + dir2)
#
# # plot('log\\one_hidden_layer_relu_crossentropy_lr=0.1_m=0_10000epochs_3')

class Plot:
    def __init__(self, maxEpoch=1000000):
        plt.clf()

        self.maxEpoch = maxEpoch
        
        dpi = 100
        self.fig, self.subplots = plt.subplots(2, 2, figsize=(1300 / dpi, 1000 / dpi), dpi=dpi)
        # fig.figure(figsize=(3.841, 7.195), dpi=100)

        # plt.subplots_adjust(left=0.1, right=0.95)

        plt.subplots_adjust(hspace=0.6, left=0.1, right=0.95)

        self.subplots[0, 0].set_title('Training Loss Per Epoch')
        self.subplots[0, 0].set_xlabel('Epochs')
        self.subplots[0, 0].set_ylabel('Loss')
        # self.subplots[0, 0].set_ylim(ymax=3.01)

        self.subplots[0, 1].set_title('Training Acc Per Epoch')
        self.subplots[0, 1].set_xlabel('Epochs')
        self.subplots[0, 1].set_ylabel('Accuracy')
        # self.subplots[0, 1].set_ylim(0.29, 1.01)

        self.subplots[1, 0].set_title('Validation Loss Per Epoch')
        self.subplots[1, 0].set_xlabel('Epochs')
        self.subplots[1, 0].set_ylabel('Loss')
        # self.subplots[1, 0].set_ylim(ymax=3.01)

        self.subplots[1, 1].set_title('Validation Acc Per Epoch')
        self.subplots[1, 1].set_xlabel('Epochs')
        self.subplots[1, 1].set_ylabel('Accuracy')
        # self.subplots[1, 1].set_ylim(0.29, 1.01)

    def addPlot(self, subplot, x, y, label, fmt):
        subplot.plot(x, y, fmt, label=label)

    def process(self):
        self.subplots[0, 0].legend(loc='best')
        self.subplots[0, 1].legend(loc='best')
        self.subplots[1, 0].legend(loc='best')
        self.subplots[1, 1].legend(loc='best')

    def show(self):
        self.process()
        plt.show()

    def save(self, filename):
        self.process()
        plt.savefig(filename)

    def addModel(self, dir, label, fmt):
        TrainEpochs, TrainLoss, TrainAcc = ReadData(dir + '/train.txt')
        ValEpochs, ValLoss, ValAcc = ReadData(dir + '/valid.txt')
        # print(dir, max(ValAcc))
        self.addPlot(self.subplots[0, 0], TrainEpochs[0 : self.maxEpoch], TrainLoss[0 : self.maxEpoch], label, fmt)
        self.addPlot(self.subplots[0, 1], TrainEpochs[0 : self.maxEpoch], TrainAcc[0 : self.maxEpoch], label, fmt)
        self.addPlot(self.subplots[1, 0], ValEpochs[0 : self.maxEpoch], ValLoss[0 : self.maxEpoch], label, fmt)
        self.addPlot(self.subplots[1, 1], ValEpochs[0 : self.maxEpoch], ValAcc[0 : self.maxEpoch], label, fmt)

plot1 = Plot(100)
model = 'CNN'
plot1.addModel('logs/%s_without_attention_droprate=0.0' % model, '%s without attention, droprate=0.0' % model, '-b')
plot1.addModel('logs/%s_without_attention' % model, '%s without attention, droprate=0.5' % model, '-k')
plot1.save('%s.png' % model)

# for root, dirs, files in os.walk("logs/"):
#     if root == "logs/":
#         for f in dirs:
#             plot1 = Plot()
#             plot1.addModel('logs/'+f, f, '-b')
#             # plot1.show()
#             # plot1.save(f.replace('.', '_'))
#             print("<img src=\"codes\%s.png\">" % f.replace('.', '_'))
# plot1.addModel('log/mlp_dropout_1000hiddennodes_droprate=0.5', 'MLP without BN', '-b')
# plot1.addModel('../cnn/log/cnn_dropout_bn_64_128', 'CNN with BN', '-r')
# plot1.addModel('../cnn/log/cnn_dropout_64_128', 'CNN without BN', '-c')
# plot1.show()
# plot1.save('../../MLPandCNN.png')

# plot2 = Plot(200)
# plot2.addModel('log/mlp_dropout_bn_1000hiddennodes_droprate=0', 'with BN, Drop Rate=0', '-b')
# plot2.addModel('log/mlp_dropout_bn_1000hiddennodes_droprate=0.1', 'with BN, Drop Rate=0.1', '-c')
# plot2.addModel('log/mlp_dropout_bn_1000hiddennodes_droprate=0.3', 'with BN, Drop Rate=0.3', '-r')
# plot2.addModel('log/mlp_dropout_bn_1000hiddennodes_droprate=0.5', 'with BN, Drop Rate=0.5', '-k')
# plot2.addModel('log/mlp_dropout_bn_1000hiddennodes_droprate=0.7', 'with BN, Drop Rate=0.7', '-m')
# plot2.addModel('log/mlp_dropout_bn_1000hiddennodes_droprate=0.9', 'with BN, Drop Rate=0.9', '-g')
# # plot2.show()
# plot2.save('../../MLPwithBN.png')
#
# plot3 = Plot(200)
# plot3.addModel('log/mlp_dropout_1000hiddennodes_droprate=0', 'without BN, Drop Rate=0', '-b')
# plot3.addModel('log/mlp_dropout_1000hiddennodes_droprate=0.1', 'without BN, Drop Rate=0.1', '-c')
# plot3.addModel('log/mlp_dropout_1000hiddennodes_droprate=0.3', 'without BN, Drop Rate=0.3', '-r')
# plot3.addModel('log/mlp_dropout_1000hiddennodes_droprate=0.5', 'without BN, Drop Rate=0.5', '-k')
# plot3.addModel('log/mlp_dropout_1000hiddennodes_droprate=0.7', 'without BN, Drop Rate=0.7', '-m')
# plot3.addModel('log/mlp_dropout_1000hiddennodes_droprate=0.9', 'without BN, Drop Rate=0.9', '-g')
# # plot3.show()
# plot3.save('../../MLPwithoutBN.png')
# #
# class Plot2:
#
#     def __init__(self):
#         dpi = 100
#         self.fig, self.subplot = plt.subplots(1, 1, figsize=(1300 / dpi, 1000 / dpi), dpi=dpi)
#         self.subplot.set_title('Test Accuracy with/without BN')
#         self.subplot.set_ylabel('Accuracy')
#         self.subplot.set_xticks([0, 1], ['without BN', 'with BN'])
#         self.subplot.tick_params(labelright=True)
#
#     def addModel(self, dir1, dir2, label, fmt):
#         TestAcc1 = ReadData(dir1 + '/test.txt')[2][-1]
#         TestAcc2 = ReadData(dir2 + '/test.txt')[2][-1]
#         self.subplot.plot([0, 1], [TestAcc1, TestAcc2], fmt, label=label)
#
#     def process(self):
#         self.subplot.legend(loc='best')
#
#     def show(self):
#         self.process()
#         plt.show()
#
#     def save(self, filename):
#         self.process()
#         plt.savefig(filename)
#
# plot4 = Plot2()
# # plot4.addModel('../cnn/log/cnn_dropout_64_128', '../cnn/log/cnn_dropout_bn_64_128', 'CNN, Drop Rate=0', 'x-y')
# plot4.addModel('log/mlp_dropout_1000hiddennodes_droprate=0', 'log/mlp_dropout_bn_1000hiddennodes_droprate=0', 'MLP, Drop Rate=0', 'x-b')
# plot4.addModel('log/mlp_dropout_1000hiddennodes_droprate=0.1', 'log/mlp_dropout_bn_1000hiddennodes_droprate=0.1', 'MLP, Drop Rate=0.1', 'x-c')
# plot4.addModel('log/mlp_dropout_1000hiddennodes_droprate=0.3', 'log/mlp_dropout_bn_1000hiddennodes_droprate=0.3', 'MLP, Drop Rate=0.3', 'x-r')
# plot4.addModel('log/mlp_dropout_1000hiddennodes_droprate=0.5', 'log/mlp_dropout_bn_1000hiddennodes_droprate=0.5', 'MLP, Drop Rate=0.5', 'x-k')
# plot4.addModel('log/mlp_dropout_1000hiddennodes_droprate=0.7', 'log/mlp_dropout_bn_1000hiddennodes_droprate=0.7', 'MLP, Drop Rate=0.7', 'x-m')
# plot4.addModel('log/mlp_dropout_1000hiddennodes_droprate=0.9', 'log/mlp_dropout_bn_1000hiddennodes_droprate=0.9', 'MLP, Drop Rate=0.9', 'x-g')
# # plot4.show()
# plot4.save('../../BN.png')



class Plot3:

    def __init__(self):
        dpi = 100
        self.fig, self.subplot = plt.subplots(1, 1, figsize=(1300 / dpi, 500 / dpi), dpi=dpi)
        self.subplot.set_title('Test Accuracy with/without BN')
        self.subplot.set_ylabel('Accuracy')
        self.subplot.set_xlabel('Model')
        self.subplot.set_ylim(0.4, 0.8)
        self.acc1 = []
        self.acc2 = []
        self.label = []

    def addModel(self, dir1, dir2, label):
        TestAcc1 = ReadData(dir1 + '/test.txt')[2][-1]
        TestAcc2 = ReadData(dir2 + '/test.txt')[2][-1]
        self.acc1.append(TestAcc1)
        self.acc2.append(TestAcc2)
        self.label.append(label)

    def process(self):
        barWidth = 0.4
        index = np.arange(self.label.__len__())
        self.subplot.bar(index, self.acc1, barWidth, color='y', label='without BN', alpha=1)
        self.subplot.bar(index + barWidth, self.acc2, barWidth, color='r', label='with BN', alpha=1)
        self.subplot.set_xticks(index + barWidth / 2)
        self.subplot.set_xticklabels(self.label)
        self.subplot.legend(loc='best')

    def show(self):
        self.process()
        plt.show()

    def save(self, filename):
        self.process()
        plt.savefig(filename)

#
# plot5 = Plot3()
# plot5.addModel('../cnn/log/cnn_dropout_64_128', '../cnn/log/cnn_dropout_bn_64_128', 'CNN\nDrop Rate=0.5')
# plot5.addModel('log/mlp_dropout_1000hiddennodes_droprate=0', 'log/mlp_dropout_bn_1000hiddennodes_droprate=0', 'MLP\nDrop Rate=0')
# plot5.addModel('log/mlp_dropout_1000hiddennodes_droprate=0.1', 'log/mlp_dropout_bn_1000hiddennodes_droprate=0.1', 'MLP\nDrop Rate=0.1')
# plot5.addModel('log/mlp_dropout_1000hiddennodes_droprate=0.3', 'log/mlp_dropout_bn_1000hiddennodes_droprate=0.3', 'MLP\nDrop Rate=0.3')
# plot5.addModel('log/mlp_dropout_1000hiddennodes_droprate=0.5', 'log/mlp_dropout_bn_1000hiddennodes_droprate=0.5', 'MLP\nDrop Rate=0.5')
# plot5.addModel('log/mlp_dropout_1000hiddennodes_droprate=0.7', 'log/mlp_dropout_bn_1000hiddennodes_droprate=0.7', 'MLP\nDrop Rate=0.7')
# plot5.addModel('log/mlp_dropout_1000hiddennodes_droprate=0.9', 'log/mlp_dropout_bn_1000hiddennodes_droprate=0.9', 'MLP\nDrop Rate=0.9')
# # plot5.show()
# plot5.save('../../BN.png')


class Plot4:

    def __init__(self):
        dpi = 100
        self.fig, self.subplot = plt.subplots(1, 1, figsize=(1300 / dpi, 500 / dpi), dpi=dpi)
        self.subplot.set_title('Test Accuracy with Drop Rate')
        self.subplot.set_ylabel('Accuracy')
        self.subplot.set_xlabel('Drop Rate')
        self.subplot.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    def addModel(self, str, dropRate, label, fmt):
        accList = []
        rateList = []
        for rate in dropRate:
            model = str % rate
            rateList.append(float(rate))
            acc = ReadData(model + '/test.txt')[2][-1]
            accList.append(acc)
        self.subplot.plot(rateList, accList, fmt, label=label)


    def process(self):
        self.subplot.legend(loc='best')

    def show(self):
        self.process()
        plt.show()

    def save(self, filename):
        self.process()
        plt.savefig(filename)

# plot6 = Plot4()
# plot6.addModel("log/mlp_dropout_1000hiddennodes_droprate=%s", ["0", "0.1", "0.3", "0.5", "0.7", "0.9"], 'MLP without BN', 'x-k')
# plot6.addModel("log/mlp_dropout_bn_1000hiddennodes_droprate=%s", ["0", "0.1", "0.3", "0.5", "0.7", "0.9"], 'MLP with BN', 'x-r')
# # plot6.show()
# plot6.save('../../DropRate.png')
#

class Plot5:

    def __init__(self):
        dpi = 100
        self.fig, self.subplots = plt.subplots(3, 2, figsize=(1300 / dpi, 1000 / dpi), dpi=dpi)
        # fig.figure(figsize=(3.841, 7.195), dpi=100)

        # plt.subplots_adjust(left=0.1, right=0.95)

        plt.subplots_adjust(hspace=0.6, left=0.1, right=0.95)

    def addPlot(self, subplot, x, y, label, fmt):
        subplot.plot(x, y, fmt, label=label)

    def process(self):
        self.subplots[0, 0].legend(loc='best')
        self.subplots[0, 1].legend(loc='best')
        self.subplots[1, 0].legend(loc='best')
        self.subplots[1, 1].legend(loc='best')
        self.subplots[2, 0].legend(loc='best')
        self.subplots[2, 1].legend(loc='best')

    def show(self):
        self.process()
        plt.show()

    def save(self, filename):
        self.process()
        plt.savefig(filename)

    def addModel(self, x, y, dir, label, fmt):
        ValEpochs, ValLoss, ValAcc = ReadData(dir + '/val.txt')
        self.addPlot(self.subplots[x, y], ValEpochs[0 : 200], ValAcc[0 : 200], label, fmt)

    def setXLabel(self, x, y, label):
        self.subplots[x, y].set_xlabel(label)

#
# plot7 = Plot5()
#
# plot7.addModel(0, 0, 'log/mlp_dropout_1000hiddennodes_droprate=0', 'without BN', 'k')
# plot7.addModel(0, 0, 'log/mlp_dropout_bn_1000hiddennodes_droprate=0', 'with BN', 'r')
# plot7.addModel(0, 1, 'log/mlp_dropout_1000hiddennodes_droprate=0.1', 'without BN', 'k')
# plot7.addModel(0, 1, 'log/mlp_dropout_bn_1000hiddennodes_droprate=0.1', 'with BN', 'r')
# plot7.addModel(1, 0, 'log/mlp_dropout_1000hiddennodes_droprate=0.3', 'without BN', 'k')
# plot7.addModel(1, 0, 'log/mlp_dropout_bn_1000hiddennodes_droprate=0.3', 'with BN', 'r')
# plot7.addModel(1, 1, 'log/mlp_dropout_1000hiddennodes_droprate=0.5', 'without BN', 'k')
# plot7.addModel(1, 1, 'log/mlp_dropout_bn_1000hiddennodes_droprate=0.5', 'with BN', 'r')
# plot7.addModel(2, 0, 'log/mlp_dropout_1000hiddennodes_droprate=0.7', 'without BN', 'k')
# plot7.addModel(2, 0, 'log/mlp_dropout_bn_1000hiddennodes_droprate=0.7', 'with BN', 'r')
# plot7.addModel(2, 1, 'log/mlp_dropout_1000hiddennodes_droprate=0.9', 'without BN', 'k')
# plot7.addModel(2, 1, 'log/mlp_dropout_bn_1000hiddennodes_droprate=0.9', 'with BN', 'r')
# plot7.setXLabel(0, 0, 'MLP, Drop Rate=0')
# plot7.setXLabel(0, 1, 'MLP, Drop Rate=0.1')
# plot7.setXLabel(1, 0, 'MLP, Drop Rate=0.3')
# plot7.setXLabel(1, 1, 'MLP, Drop Rate=0.5')
# plot7.setXLabel(2, 0, 'MLP, Drop Rate=0.7')
# plot7.setXLabel(2, 1, 'MLP, Drop Rate=0.9')
# # plot7.show()
# plot7.save('../../MLPDropRate.png')


class Plot6:

    def __init__(self):
        dpi = 100
        self.fig, self.subplots = plt.subplots(6, 2, figsize=(1300 / dpi, 2000 / dpi), dpi=dpi)
        # fig.figure(figsize=(3.841, 7.195), dpi=100)

        # plt.subplots_adjust(left=0.1, right=0.95)


        plt.subplots_adjust(hspace=0.6, left=0.1, right=0.95)
        for i in range(6):
            self.subplots[i, 0].set_xlabel('Epochs')
            self.subplots[i, 0].set_ylabel('Loss')
            self.subplots[i, 1].set_xlabel('Epochs')
            self.subplots[i, 1].set_ylabel('Accuracy')

    def addPlot(self, subplot, x, y, label, fmt):
        subplot.plot(x, y, fmt, label=label)

    def process(self):
        for i in range(6):
            for j in range(2):
              self.subplots[i, j].legend(loc='best')

    def show(self):
        self.process()
        plt.show()

    def save(self, filename):
        self.process()
        plt.savefig(filename)

    def addModel(self, x, dir, title):
        TrainEpochs, TrainLoss, TrainAcc = ReadData(dir + '/train.txt')
        ValEpochs, ValLoss, ValAcc = ReadData(dir + '/val.txt')
        self.addPlot(self.subplots[x, 0], TrainEpochs[0 : 200], TrainLoss[0 : 200], 'Train Set', 'k')
        self.addPlot(self.subplots[x, 0], ValEpochs[0 : 200], ValLoss[0 : 200], 'Validation Set', 'r')
        self.addPlot(self.subplots[x, 1], TrainEpochs[0 : 200], TrainAcc[0 : 200], 'Train Set', 'k')
        self.addPlot(self.subplots[x, 1], ValEpochs[0 : 200], ValAcc[0 : 200], 'Validation Set', 'r')
        self.subplots[x, 0].set_title(title)
        self.subplots[x, 1].set_title(title)
#
#
#
# plot8 = Plot6()
# plot8.addModel(0, 'log/mlp_dropout_1000hiddennodes_droprate=0', 'MLP without BN, Drop Rate=0')
# plot8.addModel(1, 'log/mlp_dropout_1000hiddennodes_droprate=0.1', 'MLP without BN, Drop Rate=0.1')
# plot8.addModel(2, 'log/mlp_dropout_1000hiddennodes_droprate=0.3', 'MLP without BN, Drop Rate=0.3')
# plot8.addModel(3, 'log/mlp_dropout_1000hiddennodes_droprate=0.5', 'MLP without BN, Drop Rate=0.5')
# plot8.addModel(4, 'log/mlp_dropout_1000hiddennodes_droprate=0.7', 'MLP without BN, Drop Rate=0.7')
# plot8.addModel(5, 'log/mlp_dropout_1000hiddennodes_droprate=0.9', 'MLP without BN, Drop Rate=0.9')
# # plot8.show()
# plot8.save('../../Dropout.png')