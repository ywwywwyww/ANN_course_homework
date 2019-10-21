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
    fig, subplots = plt.subplots(1, 2, figsize=(1300 / dpi, 500 / dpi), dpi=dpi)
    # fig.figure(figsize=(3.841, 7.195), dpi=100)

    plt.subplots_adjust(left=0.1, right=0.95)

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
    def __init__(self):
        dpi = 100
        self.fig, self.subplots = plt.subplots(1, 2, figsize=(1300 / dpi, 500 / dpi), dpi=dpi)
        # fig.figure(figsize=(3.841, 7.195), dpi=100)

        plt.subplots_adjust(left=0.1, right=0.95)

        plt.subplots_adjust(hspace=0.6, left=0.1, right=0.95)

        self.subplots[0, 0].set_title('Training Loss Per Epoch')
        self.subplots[0, 0].set_xlabel('Epochs')
        self.subplots[0, 0].set_ylabel('Loss')
        self.subplots[0, 0].legend(loc='best')
        self.subplots[0, 0].set_ylim(-0.01, 3.01)

        self.subplots[0, 1].set_title('Training Acc Per Epoch')
        self.subplots[0, 1].set_xlabel('Epochs')
        self.subplots[0, 1].set_ylabel('Accuracy')
        self.subplots[0, 1].legend(loc='best')
        self.subplots[0, 1].set_ylim(0.29, 0.81)

        self.subplots[1, 0].set_title('Validation Loss Per Epoch')
        self.subplots[1, 0].set_xlabel('Epochs')
        self.subplots[1, 0].set_ylabel('Loss')
        self.subplots[1, 0].legend(loc='best')
        self.subplots[1, 0].set_ylim(-0.01, 3.01)

        self.subplots[1, 1].set_title('Validation Acc Per Epoch')
        self.subplots[1, 1].set_xlabel('Epochs')
        self.subplots[1, 1].set_ylabel('Accuracy')
        self.subplots[1, 1].legend(loc='best')
        self.subplots[1, 1].set_ylim(0.29, 0.81)

    def addPlot(self, subplot, x, y, label, fmt):
        subplot.plot(x, y, label=label, fmt=fmt)

    def show(self):
        plt.show()

    def save(self, filename):
        plt.savefig(filename)

    def addModel(self, dir, label, fmt):
        TrainEpochs, TrainLoss, TrainAcc = ReadData(dir + '/train.txt')
        ValEpochs, ValLoss, ValAcc = ReadData(dir + '/val.txt')
        self.addPlot(self.subplots[0, 0], TrainEpochs, TrainLoss, label, fmt)
        self.addPlot(self.subplots[0, 1], TrainEpochs, TrainAcc, label, fmt)
        self.addPlot(self.subplots[1, 0], ValEpochs, ValLoss, label, fmt)
        self.addPlot(self.subplots[1, 1], ValEpochs, ValAcc, label, fmt)



plot1 = Plot()
plot1.addModel()
plot1.show()