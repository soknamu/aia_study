from tensorflow.keras.preprocessing.text import Tokenizer #전처리 개념.

text = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'

#Tokenizer : 단어별로 짜르겠다, 수치화를 지정 해야됨.

token =  Tokenizer()
token.fit_on_texts([text]) #text를 트레인 시킨다. 문장이 여러개가 있을수 있으니 리스트모양으로 만들어줌.

'''
print(token.word_index) 
#{'마구': 1, '매우': 2, '나는': 3, '진짜': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}
#마구가 가장 많아서 1번으로 감. 두번째는 매우2개여서 2번째,

print(token.word_counts)
#OrderedDict([('나는', 1), ('진짜', 1), ('매우', 2), ('맛있는', 1), ('밥을', 1), ('엄청', 1), ('마구', 3), ('먹었다', 1)])
# 단어 뒤에 나온 숫자는 단어의 개수.
'''

x = token.texts_to_sequences([text])
#print(x) 
#print(type(x)) <class 'list'>
#[[3, 4, 2, 2, 5, 6, 7, 1, 1, 1, 8]] = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.' ->(1, 11) 1행 11열


#그냥 계산을 하면 숫자의 수치에 가치가 있다고 판단해서 원핫 인코딩을 해줘야됨.

# ###################1. to_categorical ##################
# from tensorflow.keras.utils import to_categorical
# x = to_categorical(x)
# print(x)
# # [[[0. 0. 0. 1. 0. 0. 0. 0. 0.]
# #   [0. 0. 0. 0. 1. 0. 0. 0. 0.]
# #   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
# #   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
# #   [0. 0. 0. 0. 0. 1. 0. 0. 0.]
# #   [0. 0. 0. 0. 0. 0. 1. 0. 0.]
# #   [0. 0. 0. 0. 0. 0. 0. 1. 0.]
# #   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
# #   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
# #   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
# #   [0. 0. 0. 0. 0. 0. 0. 0. 1.]]]
# print(x.shape) # (1, 11, 9)
# #결국 0을 지우고 (11,1)로 reshape


###################2. pandas ##################
import pandas as pd
import numpy as np

x = np.array(pd.get_dummies(x[0]))

print(x) #TypeError: unhashable type: 'list'
# 오류 : 첫번째 리스트를 넘파이로 바꿔야 된다.
#        두번째 그러면 리스트는 왜 안될까?        