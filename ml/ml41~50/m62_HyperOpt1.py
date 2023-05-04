#최소값을 찾는것이다!
#베이시안옵티마이제이션은 최대값을 찾는것
import hyperopt #0.2.7
import numpy as np
from hyperopt import hp, fmin, tpe, Trials #최솟값을 찾는 함수.
#                                    히스토리역할
search_space = {
    
 'x1' : hp.quniform('x1',-10,10,1), #-10부터 10까지 1단위로
'x2' : hp.quniform('x2',-15,15,1) #quniform : q의 간격대로 정규화
#      hp.quniform(label,low,high,q)
}#                 이름,최소,최대,단위

#print(search_space) #{'x': <hyperopt.pyll.base.Apply object at 0x0000019CF5EE6220>, 'x2': <hyperopt.pyll.base.Apply object at 0x0000019CFB5A17F0>}

def objective_func(search_space):
    x1 = search_space['x1']
    x2 = search_space['x2']
    return_value = x1**2 -20*x2 #x1가 0일때,x2가 15일때 최소값. 
    
    return return_value

trials_val = Trials()

best = fmin(
    fn= objective_func,
    space= search_space,#파라미터
    algo= tpe.suggest, #디폴트
    max_evals=50, #n_iter
    trials=trials_val,
    rstate= np.random.default_rng(seed=10) #randomstate
)

print('best : ',best)
#best :  {'x': 5.0, 'x2': 8.0} 10번
#best :  {'x': -1.0, 'x2': 11.0} 20번
#best :  {'x': -1.0, 'x2': 15.0}
# best :  {'x': 0.0, 'x2': 15.0} 랜덤스테이트 10번,evals 20번
#print(trials_val.results)
#[{'loss': -216.0, 'status': 'ok'}, ... , {'loss': 0.0, 'status': 'ok'}]
#print(trials_val.vals)

##################데이터 프라임에 trial_val ##################
#pandas dataframe로 구해보기
import pandas as pd
# trials_val = pd.DataFrame(trials_val)
# print(trials_val)

results = [aaa['loss']for aaa in trials_val.results]
# for aaa in trials_val.results:
#     losses.append(aaa['loss']) 위에랑 같은 뜻.

df = pd.DataFrame({'x1' : trials_val.vals['x1'],
                   'x2' : trials_val.vals['x2'],
                   'results' : results})

print(df)