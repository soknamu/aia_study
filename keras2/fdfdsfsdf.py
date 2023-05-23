from sklearn.datasets import load_iris
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, train_test_split
import tensorflow as tf
import pandas as pd
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# Iris 데이터셋 로드
iris = load_iris()
x = iris.data
y = iris.target

# 훈련 데이터와 테스트 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=337
)

print(x_train.shape, y_train.shape)  # (120, 4) (120, 3)
print(x_test.shape, y_test.shape)  # (30, 4) (30, 3)

# 모델 생성 함수
def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=64, node2=64, node3=64, lr=0.001):
    inputs = Input(shape=(4,), name='inputs')
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(256, activation=activation, name='hidden4')(x)
    x = Dropout(drop)(x)
    outputs = Dense(3, activation='softmax', name='outputs')(x)

    model = Model(inputs=inputs, outputs=outputs)
    if type(optimizer) != str:
        optimizer = optimizer(learning_rate=lr)
    model.compile(optimizer=optimizer, metrics=['acc'],
                  loss='sparse_categorical_crossentropy')
    return model

# 하이퍼파라미터 생성 함수
def create_hyperparameters():
    batch_sizes = [100, 200, 300, 400, 500]
    optimizers = [Adam, RMSprop, Adadelta]
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    nodes = [32, 64, 128, 256]
    
    return {
        'batch_size': batch_sizes,
        'optimizer': optimizers,
        'drop': dropouts,
        'activation': activations,
        'lr': learning_rates,
        'node1': nodes,
        'node2': nodes,
        'node3': nodes,
    }

# 모델과 하이퍼파라미터 준비
hyperparameters = create_hyperparameters()
keras_model = KerasClassifier(build_fn=build_model, verbose=1)
model = RandomizedSearchCV(keras_model, hyperparameters, cv=2, n_iter=1, verbose=1)

# 훈련 및 평가
model.fit(x_train, y_train, epochs=100)

y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
print('accuracy score : ', acc)
