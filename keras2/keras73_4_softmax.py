import numpy as np
import matplotlib.pyplot as plt

# def softmax(x):
#     return np.exp(x)/ np.sum(np.exp(x)) # x1/ x1+x2+x3 랑 비슷한 거. 0~1사이에 수렴.

softmax = lambda x : np.exp(x)/ np.sum(np.exp(x))

x = np.arange(1, 100, 0.5)
y=softmax(x)

ratio = y
labels = y

plt.pie(ratio, labels, shadow=True, startangle=90)
plt.show()