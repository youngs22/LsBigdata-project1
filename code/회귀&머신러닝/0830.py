# ctrl + k + s 바로가기뜨게함
# alt + 위  해당 문장(블록) 위로감
# alt + z  짤린거 아래문장으로 내려감
# f11 전체화면
# ctrl + w 해당 창 닫기
# ctrl + alt + 오른쪽   해당 파일 오른쪽 창으로 넘어감
# ctrl + alt + 위/아래로    커서가 여러개가 생김
# ctrl + shift + p snippet 설정하기

def g(x=3):
    result = x +1
    return result
g()
print(g) 
# fn + f12 해당 커서에 해당하는거 찾아줌, 함수 내용보기 가능

# 함수 내용 확인하기
import inspect
print(inspect.getsource(g))

# if else
x=3
if x>4:
    y=1
else:
    y=2
print(y)

# if else 축약
y = 1 if x > 4 else 2
y
# 리스트 컴프리헨션
x = [1, -2, 3, -4, 5]
result = ["양수" if value > 0 else "음수" for value in x]
print(result)

# 조건이 3개 이상인 경우
x = 0
if x > 0:
    result = "양수"
elif x == 0:
    result = "0"
else:
    result = "음수"
print(result)

# 조건 3가지 넘파이 버전
import numpy as np
x = np.array([1, -2, 3, -4, 0])
conditions = [x > 0,x == 0,x < 0]
choices = ["양수","0","음수"]
result = np.select(conditions, choices, x)
print(result)

# 반복문
for i in range(1, 4):
    print(f"Here is {i}")
    
print([f"Here is {i}" for i in range(1, 4)])

name = "John"
age = 30
greeting = f"Hello, my name is {name} and I am {age} years old."
print(greeting)

import numpy as np
names = ["John", "Alice"]
ages = np.array([25, 30])

# zip() 함수로 names와 ages를 병렬적으로 묶음
zipped = zip(names, ages)

# 각 튜플을 출력
for name, age in zipped:
    print(f"Name: {name}, Age: {age}")

# While 문
i=0
while i <= 10:
    i += 3
    print(i)

# 무한 루프 
i = 0
while True:
    i += 3
    if i > 10:
        break
print(i)

# apply 함수 이해하기
import pandas as pd
data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data)
print(df)

df.apply(max, axis=0)
df.apply(max, axis=1)
df.apply(sum, axis=1)

# 함수에 apply 적용
def my_f(x, const=3):
    return max(x)**2 + const

my_f([1,2,3],5)

# 데이터 프레임에 apply 적용
df.apply(my_f, axis=1)
df.apply(my_f, axis=0)
df.apply(my_f, axis=1, const=5)

# 넘파이에 apply 적용
import numpy as np
array_2d = np.arange(1, 13).reshape((3, 4), order='F')
print(array_2d)

np.apply_along_axis(max, axis=0, arr=array_2d)

#
y = 2
def my_func(x):
    y = 1
    result = x + y
    return result
print(y)

y = 2
def my_func(x):
    global y
    y= y+1
    result = x + y
    return result
print(y)

my_func(3)

y = 2
def my_func(x):
    global y
    y=my_f(y,3)
    result = x + y
    return result
print(y)

my_func(3)

# 입력값이 몇 개 일지 모를땐 별표*를 붙임
def add_many(*args):
    result = 0
    for i in args:
        result = result + i
    return result

add_many(1,2,3)
