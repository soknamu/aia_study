#[실습] 얼리스타핑 적용하려면 어떻게 하면 될까요?
# 1. 최소값을 넣을 변수를 하나, 카운트할 변수 하나 준비!
# 2. 다음 에포에 값과 최소값을 비교, 
#    최소값이 갱신되면 그 변수에 최소값을 넣어주고, 카운트변수 초기화
# 3. 갱신이 안되면 카운트 변수 ++1
# 4. 카운트 변수가 내가 원하는 earlystopping 개수에 도달하면 for문을 stop

x = 10
y = 10
w = 11
lr = 0.1
epochs = 300000

# Initialize variables for early stopping
min_loss = float('inf')
count = 0
early_stopping_threshold = 20  # Define the desired early stopping criterion

for i in range(epochs):
    hypothesis = x * w
    loss = (hypothesis - y) ** 2  # MSE

    print('epoch:', i, '\t', 'Loss:', round(loss, 4), '\t', 'Predict:', round(hypothesis, 4))

    up_predict = x * (w + lr)
    up_loss = (y - up_predict) ** 2

    down_predict = x * (w - lr)
    down_loss = (y - down_predict) ** 2

    if up_loss >= down_loss:
        w -= lr
    else:
        w += lr

    # Early stopping check
    if loss < min_loss:
        min_loss = loss
        count = 0  # Reset the count
    else:
        count += 1

    if count >= early_stopping_threshold:
        print('Early stopping triggered at epoch', i)
        break
#Early stopping triggered at epoch 10020