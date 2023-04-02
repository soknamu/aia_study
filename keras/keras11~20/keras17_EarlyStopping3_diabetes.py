from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
#1.데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

# print(x.shape, y.shape) #(442, 10) (442,) input_dim = 10

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,  #트레인과 테스트를 분리시키는 함수.
        train_size=0.7, shuffle=True, random_state=650874)
#650874

#2. 모델구성

model = Sequential()
#model.add(Dense(12,input_dim=10))
model.add(Dense(10,input_dim=10))
model.add(Dense(30,activation='relu'))
model.add(Dense(50))
model.add(Dense(100,activation='relu'))
model.add(Dense(50))
model.add(Dense(30,activation='relu'))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일
from tensorflow.python.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor = 'val_loss', patience =20, mode = 'min',
              verbose=1,
              restore_best_weights=True) 

model.compile(loss = 'mse', optimizer= 'adam')
hist = model.fit(x_train, y_train, epochs = 5000, batch_size =32, 
                verbose = 1, validation_split= 0.2,
                callbacks= [es])

print("===================발로스===================")
print(hist.history['val_loss'])
print("===================발로스====================")


#4. 평가예측

loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize =(9,6))
#plt.plot(y) #-> x는 순서대로 가기때문에 x는 명시안해도됨
plt.plot(hist.history['loss'], marker = '.', c='red',label = 'loss') #-> 이것을 통해서 어느지점에서 loss가 줄어들고, 늘어나는지 알수 있음, 
plt.plot(hist.history['val_loss'], marker = '.', c='blue',label = 'val_loss') 
                               #또한 과적합 부분을 찾아서 줄일수도 있음.
plt.title('당뇨') #표제목
plt.xlabel('epochs') # x축 
plt.ylabel('로스, 검증로스') #y축
plt.legend() #범례표시 : 오른쪽 위에 뜨는 
plt.grid()  # 그래프에 오목판처럼 축이 생김
plt.show()