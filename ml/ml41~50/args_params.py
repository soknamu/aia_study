#여러개의 인수를 받을때 키워드 인수를 받을때 사용하는 표시.

def hello(*args): #arguments
    print(args)
    for name in args:
        print(f"안녕, {name}")
# * 여러개가 들어올지 모를 때 별한개를 사용. 변수 인수를 한방에 받겠다.

hello("홍길동", "김개똥", "연개소문")

def hello(**kwargs):
    print(kwargs)
    for (key, value) in kwargs.items():
        print(f"{key}: {value}")


hello(name="심교훈", skill="파이썬", job="개발자")

# ** key value : 딕셔너리{}
# 파라미터들을 딕셔너리 형태로 여러개로 받겠다. 