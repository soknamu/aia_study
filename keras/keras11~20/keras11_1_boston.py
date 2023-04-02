from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

print(datasets.feature_names)

# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=29)

# [실습]
# 1. train 0.7
# 2. R2 0.8 이상

# 2. 모델 구성
model = Sequential([
    Dense(64, input_dim=13, activation='relu'),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

# 3. 컴파일, 훈련
optimizer = Adam(learning_rate=0.0005)
model.compile(loss='mse', optimizer=optimizer) 
model.fit(x_train, y_train, epochs=500, batch_size=10)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print("r2 score : ", r2)