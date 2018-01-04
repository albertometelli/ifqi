import numpy as np
import matplotlib.pyplot as plt

history = np.load('../results/history_example.npy')
index = 2 #1: theta, 2: avg return, 3: gradient
plt.plot(range(history.shape[0]),history[:,index])
plt.show()
