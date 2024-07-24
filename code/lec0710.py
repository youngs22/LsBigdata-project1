## 변수개념이해와 연산들

a=1
a 
a=5
a="안녕하세요"
a='안녕하세요!'
a="'안녕하세요'라고 아빠가 말했다"
a
a='"안녕하세요"라고 아빠가 말했다'
a

a=[1,2,3]
b=[4,5,5]
a+b
c="안녕하세요!"
c+b #문자열끼리만 결합할 수 있음


## 중복되는 값 제거해서 출력
ab = list(set(a+b))
ab

c = "안녕하세요!"
d = "LS빅데이터스쿨"
c+d
c + " " +d

num1=3
num2=5
num1+num2
str(num1) + str(num2)

a=10
b=3.3
print("a + b =",a+b)
print("a - b =",a-b)
print("a * b =",a*b)
print("a / b =",a/b) #나누기
print("a & b =",a%b) #나머지
print("a // b =",a//b) #몫
print("a ** b =",a**b) #거듭제거

print("a / b =",round(a/b,4)) #round(,)사용해 소수점 자리 지정

### shift+Alt+아래화살표 : 아래로 복사
a**2
(a**3)//7
(a**3)//7 
###shift+Alt+아래화살표 : 아래로 복사
###ctrl+Alt+아래/위화살표 : 커서 여러개

### a랑 b가 같니? =등호 두개 사용
a==b
a!=b
a<b
a>=b

###2에4승과 12453을 7로 나눈 몫을 더해서 8로 나눴을 때 나머지
A=((2**4)+(12453//7))%8
###9의7승을 12로 나누고, 36452를 253으로 나눈 나머지에 곱한 수
B=((9**7)/12)*(36452%253)
A<B

user_age=int(input("나이 : "))
is_adult=user_age>=18
print("성인입니까?",is_adult)

is_adult=user_age>=18
print("성인입니까?",is_adult)

##불리언(논리연산자)
a="true"

b=TRUE #예약어 True는 아니라 TRUE는 변수가 될 수 있지만, 정의되지 않은 변수임
TRUE=50
b #예약어 True는 아니라 TRUE는 변수가 될 수 있지만, 정의되지 않은 변수임

c=true #예약어 True는 아니라 TRUE는 변수가 될 수 있지만, 정의되지 않은 변수임

d=True #예약어

## True=1 / False=0
True + True #2나옴
True + False
False + False

a=True
b=False
a and b
a or b

## and 연산자
True and False
True and True
False and False
False and False

True  * False
True  * True
False * False
False * False


## or 연산자
True or True
True or False
False or True
False or False

a=True
b=False
min(a+a,1)
min(a+b,1)
min(b+a,1)
min(b+b,1)

a=3
a+=10
a
a-=4
a
a%=3
a
a+=12
a**=2
a
a/=7
a

str1="hello! "
str1+str1
repeated_str = str1*3  #문자열 사칙연산 중 덧셈과 곱셈 사용 가능
print("repeated string:", repeated_str)

str1*2.5

###정수: int(eger)
###실수: float(double)

## 단항 연산자
x=5
+x
-x
bin(x)
bin(-x)
bin(~x)

bin(5)
bin(-5)
bin(-6)

bin(3)
bin(~3)
bin(~(-3))
bin(-3)

max(3,4)
var1=[1,2,3]
sum(var1)

### pip install pydataset -> 터미널에서 돌려야됨(파이썬 안에서 돌리면 안됨)
### !pip install pydataset -> 터미널안가고 파이썬에서 돌려도 됨

import pydataset
pydataset.data()

df = pydataset.data("AirPassengers")
df
df2= pydataset.data("cake")
df2

import seaborn
var=["a","b","c","d"]
var
seaborn.countplot(x=var)

