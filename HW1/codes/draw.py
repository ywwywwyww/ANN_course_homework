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

    def draw(self):
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

    def add_training(self, steps, loss, acc):
        self.training_set_iterations.append(steps)
        self.training_set_loss.append(loss)
        self.training_set_accuracy.append(acc)

    def add_test(self, steps, loss, acc):
        self.test_set_iterations.append(steps)
        self.test_set_loss.append(loss)
        self.test_set_accuracy.append(acc)

plot = draw_plot()
# plot.add_training(1, 1, 1)
# plot.add_training(2, 2, 2)
# plot.add_test(1, 1, 1)
# plot.add_test(2, 2, 2)
# plot.draw()