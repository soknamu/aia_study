from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)

# 2. 모델 구성
model = Sequential([
    Dense(64, input_dim=13, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='relu'),
])

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
model.fit(x_train, y_train, epochs=500, batch_size=16)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print("r2 score : ", r2)