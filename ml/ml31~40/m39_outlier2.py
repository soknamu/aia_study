import numpy as np
aaa = np.array([2,3,4,5,6,7,8,1000,-10,9,10,11,12,50]) #위치 바꿔도 찾아냄.
#만약에 14, 15개가 되면 위치값이 소수단위로 바뀜(위치가 소수점으로 나타냄)
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

# import matplotlib.pyplot as plt
# plt.boxplot(aaa)
# plt.show()

####################위치x값 ####################

# import numpy as np

# aaa = np.array([2,3,4,5,6,7,8, 1000 ,-10,9,10,11,12,50])

# def outliers(data_out):
#     quartile_1, q2, quartile_3 = np.percentile(data_out,[25,50,75])
#     iqr = quartile_3 - quartile_1   
#     lower_bound = quartile_1 - (iqr * 1.5)  
#     upper_bound = quartile_3 + (iqr * 1.5)  
#     outliers = np.where((data_out>upper_bound)|(data_out<lower_bound))
#     return data_out[outliers]

# outliers_val = outliers(aaa)
# print("이상치의 값 : ", outliers_val)
