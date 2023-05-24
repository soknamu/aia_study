
x = 10
y = 10
w = 11
lr = 0.001
epochs = 3000

for i in range(epochs):
    hypothesis = x * w
    loss = (hypothesis - y) ** 2 #mse 제곱이 붙어서
    
    print('epoch:', i,'\t','Loss :', round(loss, 4),'\t','Predict : ', round(hypothesis, 4))
    
    up_predict = x * (w + lr)
    up_loss = (y - up_predict) **2

    down_predict = x * (w - lr)
    down_loss = (y - down_predict) **2

    if(up_loss >= down_loss):
        w = w -lr
    else:
        w = w + lr