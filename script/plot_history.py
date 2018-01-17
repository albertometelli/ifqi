import numpy as np
import matplotlib.pyplot as plt

history = np.load('./history.npy')
_filter = np.load('./history_filter.npy')

fig,ax = plt.subplots()
ax.plot(np.vstack(np.array(history_climb)[:, 0])[:, 0], 'p',
            label='OnOff_climb')
ax.scatter(_filter, np.vstack(np.array(history)[:, 0])[_filter, 0],
               c='r', marker='o')
ax.set_xlabel('Iteration')
ax.set_ylabel('Parameter')
legend = ax.legend(loc='upper right')
