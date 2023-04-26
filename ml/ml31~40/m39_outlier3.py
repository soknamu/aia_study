import numpy as np

aaa = ([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
       [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]])

aaa = np.transpose(aaa)

#실습 outlier1을 이용해서 이상치를 찾아라!

def outilers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out,[25,50,75], axis=0)
    print("1사분위 : ", quartile_1) 
    print("q2 :", q2)               
    print("3사분위 : ", quartile_3)
    iqr = quartile_3 - quartile_1
    print("iqr : ", iqr)
    lower_bound = quartile_1 - (iqr * 1.5)  
    upper_bound = quartile_3 + (iqr * 1.5)  
    outliers = np.where((data_out > upper_bound) | (data_out < lower_bound))
    return list(zip(outliers[0], outliers[1])) #outliers[0],outliers[1]-> 1차원 배열

outilers_loc = outilers(aaa)
print("이상치의 위치 : ", outilers_loc)

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()

#해결책: 컬럼을 두개로 나누고 for문 사용

# a_list = [[-10,2,3,4,5,6,7,8,9,10,11,12,50],
#         [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]]

# a_list = np.transpose(a_list)
# for i in (a_list):
#     def outilers(data_out):
#         quartile_1, q2, quartile_3 = np.percentile(data_out,[25,50,75], axis=0)
#         print("1사분위 : ", quartile_1) 
#         print("q2 :", q2)               
#         print("3사분위 : ", quartile_3)
#         iqr = quartile_3 - quartile_1
#         print("iqr : ", iqr)
#         lower_bound = quartile_1 - (iqr * 1.5)  
#         upper_bound = quartile_3 + (iqr * 1.5)  
#         return  np.where((data_out > upper_bound) | (data_out < lower_bound))
#     outilers_loc = outilers(a_list)
#     print("이상치의 위치 : ", outilers_loc)


#해결책: dataframe 통째로 함수로 받아드려서 return하게 수정!!
# import pandas as pd

# a = pd.DataFrame([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
#        [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]]).transpose()

# def outilers(data_out):
#     quartile_1, q2, quartile_3 = np.percentile(data_out,[25,50,75],axis = 0)
#     print("1사분위 : ", quartile_1) #4
#     print("q2 :", q2)               #7
#     print("3사분위 : ", quartile_3) #10
#     iqr = quartile_3 - quartile_1   #6
#     print("iqr : ", iqr)
#     lower_bound = quartile_1 - (iqr * 1.5)  #-5
#     upper_bound = quartile_3 + (iqr * 1.5)  #19
#     return np.where((data_out>upper_bound)|
#                     (data_out<lower_bound)) #-5<x<19

# outilers_loc = outilers(a)
# print("이상치의 위치 : ", outilers_loc)

