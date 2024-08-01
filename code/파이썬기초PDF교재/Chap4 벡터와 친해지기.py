# 벡터
import numpy as np

#Ctrl + Shift + C : 커멘트 처리

a = np.array([1, 2, 3, 4, 5]) # 숫자형 벡터 생성
b = np.array(["apple", "banana", "orange"]) # 문자형 벡터 생성
c = np.array([True, False, True, True]) # 논리형 벡터 생성
print("Numeric Vector:", a)
print("String Vector:", b)
print("Boolean Vector:", c)

# np.array도 인덱싱 가능
type(a)
a[3]
a[2:5]
a[1:4]

# 빈 array 생성
b=np.empty(3)
b
b[0]=1
b[1]=2
b[2]=3
b
b[2]


# np.array
vec1=np.array([1,2,3,4,5])
vec1

# 구간 사이에 간격이 x인 np.array 생성, 간격은 기본값 1
vec1=np.arange(100)
vec1=np.arange(1, 100)
vec1=-np.arange(1, 100)
vec1=np.arange(-100, -20)
vec1=np.arange(1, 100.1,0.5)
vec1=np.arange(100, -100,-10)
vec1

# 구간 사이에 간격이 일정한 n개의 수 np.array 생성
l_space=np.linspace(1,10,5)
l_space

l_space2=np.linspace(0,1,5,endpoint=False)
l_space2

# 반복
repeat=np.repeat("youna",3)
type(repeat)

#tile()
tile = np.tile([1,2,3],2)
tile

# repeat vs tile
vec1=np.arange(5)
vec1
np.repeat(vec1,3)
np.tile(vec1,3)

vec1=np.arange([1,2,3,4])
vec1+vec1

max(vec1)
min(vec1)

# 35672이하 홀수들의 합은?
x=np.arange(1,35672,2)
np.arange(1,35672,2).sum() #1
sum(x) #2

#len()/ shape/ size
len(x)
x.shape

b = np.array([[1, 2, 3], [4, 5, 6]])
length = len(b) # 첫 번째 차원의 길이
shape = b.shape # 각 차원의 크기
size = b.size # 전체 요소의 개수
length, shape, size

# 브로드캐스팅
a=np.array([1,2])
b=np.array([1,2,3,4])
a + b # 길이 달라서 에러남
np.tile(a,2)+b
np.repeat(a,2)+b

a=np.array([5,6,7,8])
b=np.array([1,2,3,4])
a + b
a*b

#
b==3

# 35672보다 작은 수 중에서 7로 나눠서 나머지가 3인 숫자들의 갯수는?
k=np.arange(35672)
k
x= ((k%7)==3)
x
import numpy as np
count = np.count_nonzero(x)
count


import seaborn as sns
seaborn.countplot(data=x, x=true)    

# 2차원 배열 생성
matrix = np.array([[ 0.0, 0.0, 0.0],
[10.0, 10.0, 10.0],
[20.0, 20.0, 20.0],
[30.0, 30.0, 30.0]])

matrix.shape

vector = np.array([1.0, 2.0, 3.0])
vector.shape

result = matrix + vector
result

# 에러남
vector = np.array([1.0, 2.0, 3.0, 4.0])
result = matrix + vector
result

# 세로로 벡터 더하고 싶을 때
vector = np.array([1.0, 2.0, 3.0, 4.0]).reshape(4,1)
vector
result = matrix + vector
result

# # 예시
# (4,3) + (3,) 가능
# (4,3) + (4,) 불가능
# (4,3) + (4,1) 가능
