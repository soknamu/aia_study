import numpy as np
import matplotlib.pyplot as plt

selu = lambda x, alpha=1.67326, scale=1.0507: scale * np.where(x >= 0, x, alpha * (np.exp(x) - 1))

x = np.arange(-5, 5, 0.1)

y = selu(x)
plt.plot(x, y, c='black')
plt.grid()
plt.show()

#3_2, elu 3_3, selu 3_4 reaky_relu