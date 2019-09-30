import matplotlib.pyplot as plt
import numpy as np

plotfilename = ''

class draw_plot(object):
    def __init__(self):
        self.training_set_iterations = []
        self.training_set_accuracy = []
        self.training_set_loss = []
        self.test_set_iterations = []
        self.test_set_accuracy = []
        self.test_set_loss = []
        self.learning_rate_iterations = []
        self.learning_rate = []

    def draw(self):
        training_log_file = open(plotfilename + "training.txt", "w")
        for i in range(0, self.training_set_iterations.__len__()):
            print(self.training_set_iterations[i], self.training_set_loss[i], self.training_set_accuracy[i], file=training_log_file)
        test_log_file = open(plotfilename + "test.txt", "w")
        for i in range(0, self.test_set_iterations.__len__()):
            print(self.test_set_iterations[i], self.test_set_loss[i], self.test_set_accuracy[i], file=test_log_file)
        plt.title("Loss During Training")
        plt.xlabel("Training Iterations")
        plt.ylabel("Loss")
        plt.plot(self.training_set_iterations, self.training_set_loss, color="blue", label="Training Set Loss")
        plt.plot(self.test_set_iterations, self.test_set_loss, color="orange", label="Test Set Loss")
        plt.legend(loc='upper right')
        plt.draw()
        plt.savefig(plotfilename + "loss.png")
        plt.clf()
        plt.title("Accuracy During Training")
        plt.xlabel("Training Iterations")
        plt.ylabel("Accuracy")
        plt.plot(self.training_set_iterations, self.training_set_accuracy, color="blue", label="Training Set Accuracy")
        plt.plot(self.test_set_iterations, self.test_set_accuracy, color="orange", label="Test Set Accuracy")
        plt.legend(loc='lower right')
        plt.draw()
        plt.savefig(plotfilename + "acc.png")
        plt.clf()
        plt.title("Learning Rate During Training")
        plt.xlabel("Training Iterations")
        plt.ylabel("Learning Rate")
        plt.plot(self.learning_rate_iterations, self.learning_rate, color="orange", label="Learning Rate")
        plt.legend(loc='upper right')
        plt.draw()
        plt.savefig(plotfilename + "learning_rate.png")
        plt.clf()

    def add_training(self, iterations, loss, acc):
        self.training_set_iterations.append(iterations)
        self.training_set_loss.append(loss)
        self.training_set_accuracy.append(acc)

    def add_test(self, iterations, loss, acc):
        self.test_set_iterations.append(iterations)
        self.test_set_loss.append(loss)
        self.test_set_accuracy.append(acc)

    def add_learning_rate(self, iterations, learning_rate):
        self.learning_rate_iterations.append(iterations)
        self.learning_rate.append(learning_rate)

plot = draw_plot()
# plot.add_training(1, 1, 1)
# plot.add_training(2, 2, 2)
# plot.add_test(1, 1, 1)
# plot.add_test(2, 2, 2)
# plot.draw()