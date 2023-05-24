import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)
y = np.tanh(x)

plt.plot(x,y, c= 'black')
plt.grid()
plt.show()