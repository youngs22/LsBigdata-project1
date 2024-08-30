1+1
a=1
a
#3-1 변수만들기
var1=[1,2,3]
var2=[4,5,6]
var1+var2

str1="a"
str1
str2="text"
str2
str3="hello world"
str3
str4=["a","b","c"]
str4
str5=["Hello!","world","is","good!"]
str5
str2+str3
str2+" "+str3
str1+2

#3-2 함수 이용하기
x=[1,2,3]
x
sum(x)
max(x)
min(x)
x_sum = sum(x)
x_sum
x_max = max(x)
x_max

#3-3 패키지 활용하기
library(reticulate)
use_python('C:/DS/Python/Python312/python.exe')
sns <- import('seaborn')

#4-2 데이터 프레임 이해하기
import pandas as pd
df = pd.DataFrame({ "name" : [ '김지훈','이유진','박동현','김민지' ],
                   "english" : [90,80,60,70],
                    "math" : [50,60,100,20] } )
df
df['english']
sum(df['english'])
sum(df['math'])
sum(df['english'])/4
sum(df['math'])/4

# 혼자서해보기
#Q1
df_fruit = pd.DataFrame({"product":["사과","딸기","수박"],
                         "price":[1800,1500,3000],
                         "sales":[24,38,13]})
df_fruit

#Q2
df_fruit["price"]
sum(df_fruit["price"])
sum(df_fruit["sales"])
sum(df_fruit["price"])/3
sum(df_fruit["sales"])/3
