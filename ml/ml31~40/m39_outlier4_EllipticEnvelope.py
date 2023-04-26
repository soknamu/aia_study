import numpy as np
aaa = np.array([-10, 2, 3, 4, 5, 6, 700, 8, 9,10,11,12,50])

aaa = aaa.reshape(-1,1) #얘가 요구하는 형식이 2차원이라 1차원이면 이걸해줘야됨.

from sklearn.covariance import EllipticEnvelope #공분산 함께하는
outliers = EllipticEnvelope(contamination=.1) #전체데이터중에 몇프로를 이상치를 사용할 것이냐. ex) 1이면 10%

outliers.fit(aaa)
results = outliers.predict(aaa)
print(results)
# [ 1  1  1  1  1  1 -1  1  1  1  1  1 -1]  -1 이상치의 위치.