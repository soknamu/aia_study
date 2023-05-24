import numpy as np
import matplotlib.pyplot as plt

# def sigmoid(x):
#     return 1/ (1 + np.exp(-x))
# x에 어떤값을 넣더라도 0~1사이.

sigmoid = lambda x : 1/ (1 + np.exp(-x))

x = np.arange(-5, 5, 0.1)
print(x)
print(len(x))

y = sigmoid(x)

plt.plot(x,y, c= 'black')
plt.grid()
plt.show()