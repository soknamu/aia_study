import numpy as np
import matplotlib.pyplot as plt

class ThresholdedReLULayer:
    def __init__(self, threshold):
        self.threshold = threshold

    def forward(self, x):
        return np.where(x < self.threshold, 0, x)

threshold = 0.5  # ThresholdedReLU의 임계값

x = np.arange(-5, 5, 0.1)

thresholded_relu_layer = ThresholdedReLULayer(threshold)
y = thresholded_relu_layer.forward(x)

plt.plot(x, y, c='black')
plt.grid()
plt.show()
