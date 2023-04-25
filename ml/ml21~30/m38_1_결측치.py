# interpolation
# 결측치 처리
'''

1. 삭제(행, 열)
2. 특정 값
    1) 평균값 mean -> 정말 높은 값이 들어가면 평균값이 제대로 안나옴.
    2) 중위값 median
    3) 0 : fillna
    4) 앞의 값을 대체 frontfill(ffill)(주로 온도,시간) -> 시계열데이터
    5) 뒷값 : bfill(backfill)
    6) 특정값 : ...
    7)기타 등등
3. 보간 : interpolation -> 선형보간(model에서 predict하는 방법이랑 비슷)
4. 모델 : predict
5. 트리/부스팅 계열 : 통상 결측치, 이상치에 대해 자유롭다.(결측치 처리안해도 됨.)

'''