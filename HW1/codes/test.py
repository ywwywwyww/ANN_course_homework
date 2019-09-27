import matplotlib.pyplot as plt
import numpy as np

plt.title("Loss During Training")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.plot([1,2], [2,2], color="black", label="Training Set Loss")
plt.plot([1,2], [3,5], color="blue", label="Test Set Loss")
plt.legend(loc='lower right')
plt.draw()
plt.savefig("loss.png")