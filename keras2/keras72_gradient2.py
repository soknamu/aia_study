import numpy as np
import matplotlib.pyplot as  plt

f = lambda x: x**2 -4*x +6

gradict = lambda x : 2*x -4 

x = -1000
epochs = 100
learning_rate = 0.025

print("epoch\t x\t\t f(x)")
print("{:02d}\t {:6.5f}\t {:6.5f}\t".format(0,x,f(x)))

x_history = []  # x 값 저장
f_history = []  # f(x) 값 저장

for i in range(epochs):
    x_history.append(x)
    f_history.append(f(x))
    x = x - learning_rate * gradict(x)
    # print(i+1,'\t',x,'\t', f(x))
    print("{:02d}\t {:6.5f}\t {:6.5f}\t".format(i+1,x,f(x)))
# 그래프 그리기
plt.plot(x_history, f_history, 'ro-')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent')
plt.grid(True)
plt.show()
