import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

#1. 데이터

path = './_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)

print(train_csv.shape)
print(test_csv.shape)

print(train_csv.isnull().sum()) # 결측치 없음


x = train_csv.drop(['count','casual','registered'],axis = 1)

y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x,y,
shuffle=True, random_state= 5352 , train_size=0.7)

#마이너스가 뜨는 이유 :임의의 랜덤값이 들어감(음수도 들어감) 그래서 결과값이 마이너스가 나옴.
#dense 에서 음수가 나오니깐 여기서 조정해야됨. sol) activation(활성화 함수)를 사용하면 됨 (정확도↑)
#다음으로 전달되는 값을 범위를 정해서 한정함. (Relu : 0이상의 값은 쭉올라가고, 0이하면 0으로 쭉고정, 그러면 양수가됨)
#활성화함수가 아예없지 않다.

#2.모델구성

model = Sequential()
model.add(Dense(120,input_dim = 8))
model.add(Dense(100,activation = 'relu'))
model.add(Dense(90))
model.add(Dense(80))
model.add(Dense(60,activation = 'relu'))
model.add(Dense(40))
model.add(Dense(30,activation = 'linear')) # 디폴트값. linear는 있으나 마나. #케라스 1번 문제
model.add(Dense(10,activation = 'relu'))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4,activation = 'relu'))
model.add(Dense(1))

#3. 컴파일

model.compile(loss = 'mse', optimizer ='adam')
model.fit(x_train,y_train, epochs = 4000, batch_size =300, verbose =1)

#사람이 훈련과 테스트를 나누었다. 수능 & 모의고사(문제집) ->
#지금까지 한 행동 : 교과서를 보고, 바로 수능을 침 (모의고사를 안풀고 가서 성능이 떨어짐)
#공부하고, 모의고사보고 공부하고, 모의고사보고, 수능보고 이렇게함.

#앞으로 훈련.fit에 검증(verification)을 한다. 

#4. 평가 예측

loss = model.evaluate(x_test,y_test)
print('loss : ',loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict)

print('r2 score :', r2)


def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)

print('rmse : ', rmse)

# 파일 만들기
y_submit = model.predict(test_csv)
# -> 예측 한다. test_csv를
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col = 0)

submission['count'] = y_submit

submission.to_csv(path_save + 'submit_0307_1455 .csv')

# 이름 하나하나 바꾸기 귀찮은데 변수나 함수를 적어서 실행할때마다 이름을 바꾸게 할 수는 없을까?

# r2 score : 0.24620563886133384
# rmse :  158.04622047731374
# 203/203 [==============================] - 0s 683us/step

# loss :  0.0
# 103/103 [==============================] - 0s 697us/step
# r2 score : 0.2568821313882207
# rmse :  156.9229719229334

# loss :  22247.9140625
# 103/103 [==============================] - 0s 798us/step
# r2 score : 0.3286114758522214
# rmse :  149.1573398266268

# loss :  22123.69921875
# 103/103 [==============================] - 0s 712us/step
# r2 score : 0.33235978022564594
# rmse :  148.74039076600755

# loss :  22264.3828125
# 103/103 [==============================] - 0s 885us/step
# r2 score : 0.3330504226801513
# rmse :  149.21255279316478 random = 777


