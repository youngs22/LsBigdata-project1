# 리스트 예제
fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "apple", 3.5, True]

print("과일 리스트:", fruits)
print("숫자 리스트:", numbers)
print("혼합 리스트:", mixed)


# 빈 리스트 생성
empty_list1 = []
empty_list2 = list()
print("빈 리스트 1:", empty_list1)
print("빈 리스트 2:", empty_list2)

# 초기값을 가진 리스트 생성
numbers = [1, 2, 3, 4, 5]
range_list = list(range(5))
print("숫자 리스트:", numbers)
print("range() 함수로 생성한 리스트:", range_list)

range_list[3] = "LS 빅데이터 스쿨"
range_list
range_list[2] = ["1st","2nd","3rd"]
range_list

# "3rd"만 갖고 오고 싶다면?
range_list[2][2]


# 리스트 내포(comprehension)
squares = [x**2 for x in range(10)]
squares 

squares = [x**2 for x in [3,5,2,15]] # 리스트 가능
squares 

import numpy as np
squares = [x**2 for x in np.array([3,5,2,15])] # numpy array도 가능
squares 

import pandas as pd
exam=pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/exam.csv")
exam_squares = [x**3 for x in exam["math"]] #판다스 데이터 프레임도 가능
exam_squares

#리스트 합치기
"안녕"+"하세요"
"안녕"*3

list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined_list = list1 + list2

(list1*3)+(list2*5)

# 리스트 반복
numbers = [1, 2, 3]
repeated_list = numbers * 3

# 리스트 각 원소별 반복
numbers = [5, 2, 3]
repeated_list = [x for x in numbers for _ in range(3)]
repeated_list = [x for x in numbers for _ in [1,1]]
repeated_list

# _의 의미
# 1. 앞에 나온 값을 가리킴
5+4
_+6 # _는 9를 의미

# 값 생략, placeholder

a, _, b =(1,2,4)
a; b
# del _
# _=None


# 리스트 하나 만들어서 for루프를 사용해 2,4,6,8, ..., 20 수 채우기
list2 = []
for x in range(1,11) :
   list2.append(x*2)
   
my_list = [x*2 for x in range(1,11)]   
   
   
my_list = [x*2 for x in range(1,11) for _ in range(2)]

list2 = []
for x in range(1,11) :
    for i in range(2) :
        list2.append(x*2)
        
list2 = []
for x in range(1,11) :
    for _ in range(2) :
        list2.append(x*2)
        

# for문 연습
my_list = [0]*10
for i in range(10):
    my_list[i] = 2*(i+1)

# 인덱스 공유하기 
my_listb = [2,4,6,80,10,12,24,35,23,20]
for i in range(10):
    my_list[i]=my_listb[i]

#my_listb의 홀수번째 위치에 있는 숫자들만 my_list에 갖고오기
for i in range(5):
        my_list[(i*2)-1]=my_listb[(i*2)-1]
        
# 리스트 컴프리헨션으로 바꾸는 방법
#바깥은 무조건 대괄호로 묶어줌 : 리스트를 반환하기위해
# for 루프의 :은 생략한다
# 실행부분을 먼저 써준다
# 결과값만 써준다
[i*2 for i in range(1,11)]
[x for x in numbers]

for i in range(5):
    print("hello")

for i in [0,1]:
    for j in [5,6,7]:
        print("i:",i,"j:",j)
        print(i+j)
        

for i in range(5):
    for j in range(3):
        print("i:",i,"j:",j)
        print(i+j)

# 리스트 텀프리헨션 변환
numbers = [1, 2, 3]
repeated_list = [x for x in numbers for i in range(3)]
# numbers리스트가 반복되는 것이 아닌 각 원소가 반복됨

# 원소 체크
fruits = ["apple","banana","cherry"]
"banana" in fruits
mylist=[]
for i in fruits :
    mylist.append(i == "banana")

# 바나나의 위치를 뱉어내게함
# fruits가 리스트이므로 np.array쓰려면 바꿔야됨
fruits = np.array(fruits)
np.where(fruits == "banana")[0]
int(np.where(fruits == "banana")[0][0])


# 리스트 메소드
# reverse() : 원소 뒤집기 
fruits.reverse()

# append() : 원소 맨 뒤에 원소 추가 
fruits.append("pineapple")
# 원소 맨앞에 추가하고 싶다면, reverse 후 append하고 다시 reverse

# insert : 원소 삽입
fruits.insert(2,"test")

# remove() : 원소 제거
fruits.remove("test")

# 
import numpy as np
# 넘파이 배열 생성
fruits = np.array(["apple", "banana", "cherry", "apple", "pineapple"])
# 제거할 항목 리스트
items_to_remove = np.array(["banana", "apple"])

# 불리언 마스크(논리형 벡터) 생성
mask = ~np.isin(fruits, items_to_remove)

# 불리언 마스크를 사용하여 항목 제거
filtered_fruits = fruits[mask]


