from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

datasets = load_boston()

#1. 데이터

x = datasets.data
y = datasets['target']

#print(x.shape,y.shape) #(506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x,y, 
        shuffle =True, random_state= 5253, test_size = 0.2)

#2. 모델구성

model = Sequential()
model.add(Dense(55,activation= 'relu',input_dim= 13))
model.add(Dense(44,activation= 'relu'))
model.add(Dense(33,activation= 'relu'))
model.add(Dense(22,activation= 'relu'))
model.add(Dense(11,activation= 'relu'))
model.add(Dense(2,activation= 'relu'))
model.add(Dense(1,activation= 'linear'))

#sigmoid 시그모이드

#3. 컴파일
from tensorflow.python.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor = 'val_loss', patience =20, mode = 'min',
              verbose=1,
              restore_best_weights=True, 
              
              ) 
#val_loss로 기준으로 한다. patience=5 더 추출할 것이다. 
#val_loss 의 최소값을 찾아라. R2는 여기에 없음. #auto 도 있음.(지표가 낮은게 중요하면 자동으로 낮은걸로 조정됨.)
# -> 80프로의 트레인 데이터중에서 20프로를 validation 으로 사용한다. (test 아님)
# 모델fit에서 loss값과 val_loss에서 값을 반환값(return) 그것을 반환하는 것을 hist라고함.
#print(hist) #<tensorflow.python.keras.callbacks.History object at 0x000002034FE81310> 이런 데이터의 형태다
# print(hist.history)
# restore_best_weights=True(디폴트가 false) :최저점(최상의 가중치)을(를) 잡은 지점에서 가중치가 저장됨.
model.compile(loss = 'mse', optimizer= 'adam')
hist = model.fit(x_train, y_train, epochs = 5000, batch_size =32, 
                verbose = 1, validation_split= 0.2,
                callbacks= [es]) #앞으로 콜백에서 더들어갈 수 있다. list라서.

# print("==========================================")
# print(hist)
# #<tensorflow.python.keras.callbacks.History object at 0x000001950A402820>
# print(hist.history) #hist에 대한 결과치
# print("==========================================")
# print(hist.history['loss'])
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
plt.title('보스턴') #표제목
plt.xlabel('epochs') # x축 
plt.ylabel('로스, 검증로스') #y축
plt.legend() #범례표시 : 오른쪽 위에 뜨는 
plt.grid()  # 그래프에 오목판처럼 축이 생김
plt.show()



