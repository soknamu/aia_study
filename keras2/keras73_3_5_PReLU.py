import numpy as np
import matplotlib.pyplot as plt

class PReLULayer:
    def __init__(self, alpha):
        self.alpha = alpha

    def forward(self, x):
        return np.maximum(0, x) + self.alpha * np.minimum(0, x)

alpha = 0.2  # PReLU의 alpha 값

x = np.arange(-5, 5, 0.1)

prelu_layer = PReLULayer(alpha)
y = prelu_layer.forward(x)

plt.plot(x, y, c='black')
plt.grid()
plt.show()
