from sklearn.datasets import load_iris
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
from tensorflow.keras.optimizers import Adam,RMSprop, Adadelta
tf.random.set_seed(337)
x,y = load_iris(return_X_y=True)

# y=pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, )

print(x_train.shape, y_train.shape) #(112, 4) (112,3)
print(x_test.shape, y_test.shape) #(38, 4) (38,3)

#2. 모델
def build_model(drop=0.5, optimizer= 'adam',activation='relu', 
                node1=64, node2=64, node3=64, lr=0.001):
    inputs = Input(shape=(4), name= 'inputs')
    x = Dense(node1, activation=activation, name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name = 'hidden3')(x)
    x = Dropout(drop)(x)    
    x = Dense(256, activation=activation, name = 'hidden4')(x)
    x = Dropout(drop)(x)
    outputs = Dense(3, activation='softmax', name = 'outputs')(x)

    model = Model(inputs = inputs, outputs = outputs)
    if type(optimizer)!=str:
        optimizer=optimizer(learning_rate = lr)
    model.compile(optimizer=optimizer, metrics=['acc'],
                  loss = 'sparse_categorical_crossentropy')
    return model

def create_hyperparameter():
    batchs = [100, 200, 300, 400, 500]
    optimizers = [Adam, RMSprop, Adadelta]
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    nodes = [32, 64, 128, 256]
    
    return {
        'batch_size': batchs,
        'optimizer': optimizers,
        'drop': dropouts,
        'activation': activations,
        'lr': learning_rates,
        'node1': nodes,
        'node2': nodes,
        'node3': nodes,
    }

hyperparameters = create_hyperparameter()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

keras_model = KerasClassifier(build_fn = build_model, verbose=1)
model = RandomizedSearchCV(keras_model, hyperparameters, cv=2, n_iter=1, verbose=1)

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(monitor='acc', patience=50, mode='max',verbose=1)
mcp = ModelCheckpoint(monitor='acc', mode = 'auto',
                      verbose=1,
                      save_best_only=True,
                      filepath='./_save/MCP/keras짱짱맨iris.hdf5')

import time

start = time.time()
model.fit(x_train, y_train, epochs=1000,
          callbacks = [es, mcp])
end = time.time()

print("걸린시간 : " ,end -start)
print("Best Score: ", model.best_score_) #train데이터의 최고의 스코어
print("Best Params: ", model.best_params_)
# print("Best estimator", model.best_estimator_)
print("model Score: ", model.score(x_test,y_test)) #test의 최고 스코어

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print('acc :', accuracy_score(y_test,y_predict))