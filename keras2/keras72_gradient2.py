import numpy as np
import matplotlib.pyplot as  plt

f = lambda x: x**2 -4*x +6

gradict = lambda x : 2*x -4 

x = -10
epochs = 30
learning_rate = 0.25

x_history = []  # x 값 저장
f_history = []  # f(x) 값 저장

for i in range(epochs):
    x_history.append(x)
    f_history.append(f(x))
    x = x - learning_rate * gradict(x)
    print(i, x, f(x))
    
# 그래프 그리기
plt.plot(x_history, f_history, 'ro-')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent')
plt.grid(True)
plt.show()
