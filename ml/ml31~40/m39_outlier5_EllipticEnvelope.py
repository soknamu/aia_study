import numpy as np
from sklearn.covariance import EllipticEnvelope

aaa = np.array([[-10, 2, 3, 4, 5, 6, 700, 8, 9, 10, 11, 12, 50],
               [100, 200, -30, 400, 500, 600, 700, -70000, 
                800, 900, 1000, 210, 420]]
               )
aaa = np.transpose(aaa)

outliers = EllipticEnvelope(contamination=.1,
                            #store_precision=False
                            #assume_centered=False
                            #support_fraction=21
                            )

outliers.fit(aaa)
results = outliers.predict(aaa)
results_2d = np.reshape(results, (-1, 1))

print(results_2d)

# 각 열에 대한 결측치 여부 확인
# print(np.isnan(aaa).any(axis=0))

# # 0번째 열의 결측치 개수 확인
# print(np.isnan(aaa[:, 0]).sum())

# # 1번째 열의 결측치 개수 확인
# print(np.isnan(aaa[:, 1]).sum())



#################################################################
# import numpy as np
# from sklearn.covariance import EllipticEnvelope

# aaa = np.array([[-10,2,3,4,5,6,700,8,9,10,11,12,50],
#                [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]])

# outliers = EllipticEnvelope(contamination=.1)

# for i, column in enumerate(aaa): 
#     outliers.fit(column.reshape(-1, 1))  
#     results = outliers.predict(column.reshape(-1, 1))
#     outliers_save = np.where(results == -1) # -1 인 값 위치 반환.
#     # print(outliers_save)
#     # print(outliers_save[0])
#     outliers_values = column[outliers_save] 
    
#     print(f"{i+1}번째 컬런의 이상치 : {', '.join(map(str, outliers_values))}\n 이상치의 위치 : {', '.join(map(str, outliers_save))}")
