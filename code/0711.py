#교재 63페이지

# pip install pydataset -> 터미널에서 돌려야됨(파이썬 안에서 돌리면 안됨)
# !pip install pydataset -> 터미널안가고 파이썬에서 돌려도 됨


# 패키지 활용
import pydataset
pydataset.data()

df = pydataset.data("AirPassengers")
df
df2= pydataset.data("cake")
df2

import seaborn as sns
import matplotlib.pyplot as plt

var=["a","a","c","d"]
seaborn.countplot(x=var)
plt.show()

df=sns.load_dataset("titanic")
df

plt.clf()  # 이전 데이터 지우기
sns.countplot(data=df,x="sex")
plt.show()

plt.clf() # 이전 데이터 지우기
sns.countplot(data=df,x="class")
plt.show()

plt.clf() # 이전 데이터 지우기
sns.countplot(data=df,y="class", color="red")
plt.show()

plt.clf() # 이전 데이터 지우기
sns.countplot(data=df,x="class", hue ="sex") #hue: 항목 별로 다르게 색 표시
plt.show() # dodge 변수는 기본 값이 True, 막대그래프 옆으로 독립적으로 표현

plt.clf()
sns.countplot(data=df, x='class', palette='pastel', hue='sex', dodge=False)
plt.show()


# 모듈 활용
import sklearn.metrics

import sklearn.metrics.accuracy_score()

from sklearn import metrics
metrics.accuracy_score()

from sklearn.metrics import accuracy_score
accuracy_score()

import sklearn.metrics as met
met.accuracy_score()
