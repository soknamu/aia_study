#MICE(Mutiple Imputation by Chained Equations)

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from impyute.imputation.cs import mice

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan,  8, np.nan]]
                    ).transpose()


data.columns = ['x1','x2','x3','x4']

# inpute_df = mice(data.values) 
# print(inpute_df)
#AttributeError: 'DataFrame' object has no attribute 'as_matrix' :mice에서는 판다스가 안먹힘.(numpy형태로 넣어줘야됨.)

# 1. 해결책
# inpute_df = mice(data.values) # .values 넘파이로 바꾸는 방법.
# print(inpute_df)
#선형방식으로 찾고있음. 컬럼관의 상관관계를 찾음.
# 2. 해결책
# inpute_df = mice(data.to_numpy())
# print(inpute_df)


inpute_df = mice(data.values)
print(inpute_df)

#         과제
# 1. pandas(DataFrame) -> numpy

# 2. numpy -> pandas(DataFrame)

# 3. list -> numpy

# 4. list -> pandas(DataFrame)

#각각 예제 1개씩 만들어서 제출.