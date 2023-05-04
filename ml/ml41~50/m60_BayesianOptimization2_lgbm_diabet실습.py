from bayes_opt import BayesianOptimization
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb

x,y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=337)

def lgb_cv(learning_rate, max_depth, num_leaves, 
           min_child_samples, min_child_weight, subsample, 
           colsample_bytree, reg_lambda, reg_alpha, max_bin):
    
    # LightGBM 모델의 하이퍼파라미터 설정
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'learning_rate': learning_rate,
        'max_depth': int(round(max_depth)),
        'num_leaves': int(round(num_leaves)),
        'min_child_samples': int(round(min_child_samples)),
        'min_child_weight': min_child_weight,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'max_bin': int(round(max_bin)),
        'reg_lambda': reg_lambda,
        'reg_alpha': reg_alpha,
        'device': 'gpu'
    }
    
    # LightGBM 모델 학습
    model = lgb.LGBMRegressor(**params)
    model.fit(x_train, y_train, 
              eval_set=[(x_test, y_test)], 
              early_stopping_rounds=20, 
              verbose=0)
    
    # 검증 데이터에 대한 MAE 계산
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)

    return -r2  # 목적 함수의 반환값은 음수의 MAE 값입니다. BayesOpt는 최대화를 수행하므로 음수를 취해 최소화 문제로 변환합니다.

param = {
    'learning_rate': (0.01, 0.3),
    'max_depth' : (3, 16),
    'num_leaves': (64, 256),
    'min_child_samples' : (10, 200),
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'max_bin' : (10, 500),
    'reg_lambda' : (0.001, 10),
    'reg_alpha' : (0.01, 50),
}

optimizer = BayesianOptimization(
    f = lgb_cv,
    pbounds = param,
    random_state = 337,
    verbose = 2
)

optimizer.maximize(init_points=10, n_iter=5)
#print("최적 하이퍼파라미터:", round(optimizer.max["params"],3)) #딕셔너리 형태라 에러뜸.
print("최적 하이퍼파라미터:")
for key, value in optimizer.max["params"].items():
    print(f"{key}: {value:.3f}")

print("목적 함수의 최댓값(음수의 r2):", optimizer.max["target"])

# 최적 하이퍼파라미터:
# colsample_bytree: 0.836
# learning_rate: 0.291
# max_bin: 327.613
# max_depth: 4.659
# min_child_samples: 179.489
# min_child_weight: 1.035
# num_leaves: 81.579
# reg_alpha: 41.220
# reg_lambda: 2.906
# subsample: 0.734
# 목적 함수의 최댓값(음수의 r2): 0.0010459386173313767