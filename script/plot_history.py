import numpy as np
import matplotlib.pyplot as plt

history = np.load('history.npy')
index = 1 #1: theta, 2: avg return, 3: gradient
plt.plot(range(history.shape[0]),history[:,index])
plt.show()
