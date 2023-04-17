# x = 10
# y = 'Hello, world!'
# print(type(x)) #<class 'int'>
# print(type(y)) #<class 'str'>

# a, b, c = map(int, input('-10', '20', '30').split())

# print(a + b + c)

# a,b =input('숫자 두개를 입력하세요: ').split()

# print(a)
# print(b)

# x,y,z = 10,20,30
# print(x)

# a = 10
# print(a+20)

# >>> input()
# Hello, world
# 'Hello, world'
# >>> x = input('문자열을 입력하세요:')
# 문자열을 입력하세요:Hello, world!
# >>> x
# 'Hello, world!'

# >>> a = input('첫번째 숫자를 입력하세요:')
# 첫번째 숫자를 입력하세요:10
# >>> b = input('두번째 숫자를 입력하세요:')
# 두번째 숫자를 입력하세요:20
# >>> a+b
# '1020'
# >>> print(a+b)
# 1020

# > 숫자로 인식한게 아니라, 문자로 인식해서 1020으로 인식


#그래서 문자형태를 숫자로 인식하게 만들어야되어서 int사용
# >>> a = int(input('첫번째 숫자를 입력하세요:'))
# 첫번째 숫자를 입력하세요:10
# >>> b = int(input('두번째 숫자를 입력하세요:'))
# 두번째 숫자를 입력하세요:20
# >>> a+b
# 30

# 변수 2개넣는 방법(.split()사용)
# >>> a,b = input('문자열 두 개를 입력하세요:').split()
# 문자열 두 개를 입력하세요:Hello Python
# >>> print(a)
# Hello
# >>> print(b)
# Python