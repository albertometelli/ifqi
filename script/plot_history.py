import numpy as np
import matplotlib.pyplot as plt

history = np.load('../results/history_example.npy')
index = 0 #0: theta, 1: avg return, 2: gradient
plt.plot(range(history.shape[0]),history[:,index])
plt.show()
