from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score
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

#3. 컴파일
model.compile(loss = 'mse', optimizer= 'adam')
hist = model.fit(x_train, y_train, epochs = 10, batch_size =32, 
                verbose = 1, validation_split= 0.2)
# -> 80프로의 트레인 데이터중에서 20프로를 validation 으로 사용한다. (test 아님)
# 모델fit에서 loss값과 val_loss에서 값을 반환값(return) 그것을 반환하는 것을 hist라고함.
#
#print(hist) #<tensorflow.python.keras.callbacks.History object at 0x000002034FE81310> 이런 데이터의 형태다
# print(hist.history) 

# {'loss': [589.3839721679688, 476.0315856933594, 279.8111572265625, 140.158203125, 
#           117.71404266357422, 95.91230010986328, 83.45921325683594, 76.3400650024414, 
#           70.66398620605469, 67.95389556884766, 67.68516540527344, 64.53520202636719, 
#           63.803768157958984, 62.87837600708008, 62.34392547607422, 62.0987434387207, 
#           61.71685028076172, 61.83658981323242, 61.93870162963867, 61.290889739990234, 
#           60.683780670166016, 60.14375305175781, 59.97833251953125, 60.0409049987793, 
#           61.7440071105957, 61.77470016479492, 61.380435943603516, 61.404273986816406, 
#           59.41524887084961, 58.98606872558594],

# 'val_loss': [442.6292419433594, 292.0112609863281, 109.80992126464844, 111.22956085205078, 
#               73.146728515625, 64.8846664428711, 62.918792724609375, 61.63047790527344, 58.3856086730957, 
#               57.27434158325195, 60.115047454833984, 56.362003326416016, 57.77314758300781, 59.64019775390625, 
#               59.611900329589844, 58.75884246826172, 58.152652740478516, 57.12084197998047, 58.9599494934082, 
#               56.30897521972656, 57.797935485839844, 57.379676818847656, 57.680397033691406, 61.51506423950195, 
#               55.77293014526367, 57.94905090332031, 58.3338623046875, 55.27000045776367, 56.022369384765625, 
#               59.231136322021484]}
print("==========================================")
print(hist)
#<tensorflow.python.keras.callbacks.History object at 0x000001950A402820>
print(hist.history) #hist에 대한 결과치
print("==========================================")
print(hist.history['loss'])
print("==========================================")
print(hist.history['val_loss'])
print("==========================================")


matplotlib.rcParams['font.family'] = 'malgun gothic'

#가급적이고 나눔채사용


'''
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
'''

#통상적으로 val_loss 가 loss보다 성능이 안좋음. val_loss를 위주로 봐야됨.



'''
#4. 평가예측

loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)
'''