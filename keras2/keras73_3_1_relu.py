import numpy as np
import matplotlib.pyplot as plt

# def relu(x):
#     return np.maximum(0, x) #0을 기준으로 큰값을 구하는 함수 ex)-1이면 0, 10이면 10

relu = lambda x: np.maximum(0, x)
x = np.arange(-5, 5, 0.1)

y = relu(x)
plt.plot(x,y, c= 'black')
plt.grid()
plt.show()

#3_2, elu 3_3, selu 3_4 reaky_relu