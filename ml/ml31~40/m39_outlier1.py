#이상치 탐지하는 함수.

# import numpy as np
# aaa = np.array([-10,2,3,4, 5,6, 7, 8,9, 10,11,12,50])

# def outilers(data_out):
#     str, quartile_1, q2, quartile_3,fin = np.percentile(data_out,[0,25,50,75,100]) #범위 0~100가능. percentile퍼센트위치를 나타내는 함수.
#     print("0사분위 :", str)
#     print("1사분위 : ", quartile_1) #1사분위수(25%이내)
#     print("q2 :", q2) #중앙값        (50%이내)
#     print("3사분위 : ", quartile_3) #(75프로 이내)
#     print("fin :", fin)
#     iqr = fin - str
#     print("iqr : ", iqr)
#     lower_bound = str - (iqr * 1.5)
#     upper_bound = fin + (iqr * 1.5)
#     return np.where((data_out > upper_bound) | (data_out < lower_bound)) # | : or랑 비슷(~이면 ~이값을 넣어라.) @위치를 찾아주니 걱정 x 
#     # return np.where((data_out>upper_bound)
#     #                 (data_out<lower_bound))

# outilers_loc = outilers(aaa)
# print("이상치의 위치 : ", outilers_loc)

# 1사분위 :  4.0
# q2 : 7.0
# 3사분위 :  10.0
# iqr :  6.0
# 이상치의 위치 :  (array([ 0, 12], dtype=int64),)

import numpy as np
aaa = np.array([-10,2,3,4,5,6,7,8,9,10,11,12,50])

def outilers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out,[25,50,75])
    print("1사분위 : ", quartile_1) #4
    print("q2 :", q2)               #7
    print("3사분위 : ", quartile_3) #10
    iqr = quartile_3 - quartile_1   #6
    print("iqr : ", iqr)
    lower_bound = quartile_1 - (iqr * 1.5)  #-5
    upper_bound = quartile_3 + (iqr * 1.5)  #19
    return np.where((data_out>upper_bound)|
                    (data_out<lower_bound)) #-5<x<19

outilers_loc = outilers(aaa)
print("이상치의 위치 : ", outilers_loc)

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()

#컬럼 이상치 확인
# def find_outliers(data, column, threshold=1.5):
#     outliers = []
#     q1 = np.percentile(data[column], 25)
#     q3 = np.percentile(data[column], 75)
#     iqr = q3 - q1
#     lower_fence = q1 - threshold * iqr
#     upper_fence = q3 + threshold * iqr
    
#     for i in range(len(data[column])):
#         if data[column][i] < lower_fence or data[column][i] > upper_fence:
#             outliers.append(data[column][i])
            
#     return outliers

# outliers = find_outliers(train_csv, 'Age', threshold=1.5)
# print("Outliers:", outliers)