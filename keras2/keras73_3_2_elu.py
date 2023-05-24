import numpy as np
import matplotlib.pyplot as plt

# def relu(x):
#     return np.maximum(0, x) #0을 기준으로 큰값을 구하는 함수 ex)-1이면 0, 10이면 10

import math
elu = np.vectorize(lambda x, alpha=1.0: x if x >= 0 else alpha * (math.exp(x) - 1))

x = np.arange(-5, 5, 0.1)
y = elu(x)

plt.plot(x, y, c='black')
plt.grid()
plt.show()

#3_2, elu 3_3, selu 3_4 reaky_relu