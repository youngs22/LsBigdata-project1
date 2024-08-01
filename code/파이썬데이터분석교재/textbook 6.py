import pandas as pd
import numpy as np

# 데이터 전처리 함수
# query()
# df[]

exam = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/exam.csv")
exam
# query() 조건에 맞는 행 걸러냄
exam.query("nclass==1")
exam.query("nclass!=1")
# exam[exam["nclass"]==1]
exam.query("math>50")
exam.query("math>30 & math<60")
exam.query("math>50 & nclass ==1")
exam.query("english>50 | nclass ==1")
exam.query("english>90 | math>90 | science>90")
exam.query("english>90 & math>90 & science>90")
exam.query("nclass in [1,5]")
exam.query("nclass not in [1,5]")

nclass1 = exam.query("nclass==1")
nclass1.mean()

exam.drop(columns=["math", "english"])
exam.head()

exam.sort_values("math")
exam.sort_values("math", ascending=False)
exam.sort_values(["nclass","math"], ascending=[True,False])


exam = exam.assign(
    total = exam["math"] + exam["english"] + exam["science"],
    mean = (exam["math"] + exam["english"] + exam["science"]/3))\
    .sort_values("total")
exam

exam = exam.assign(
    total = exam["math"] + exam["english"] + exam["science"])\
    .sort_values("total",ascending=False)
    
exam2 = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/exam.csv")
exam2

# lambda 사용하면, assign안에서 만든 변수(total)를 사용할 수 있음(원래는 못사용)
exam2=exam2.assign(
    total = lambda x: x["math"] + x["english"] + exam["science"],
    mean = lambda x: x["total"]/3)\
    .sort_values("total", ascending = False)
exam2


#
exam.agg(mean_math = ("math", "mean"))
exam2.groupby("nclass")\
    .agg(
        mean_math = ("math","mean"),
        sum_math = ("math", "sum"),
        median_math = ("math", "median"),
        n = ("nclass", "count")
    )

exam2
import pydataset
df = pydataset.data("mpg")
df

df.groupby(["manufacturer","drv"])\
.agg(mean_cty = ("cty","mean"))

df.query("manufacturer == 'audi'")\
    .groupby(["drv"])\
    .agg(n=("drv","count"))
    
df.query("manufacturer == 'chevrolet'")\
    .groupby(["drv"])\
    .agg(n=("drv","count"))

#
df.groupby(["manufacturer","drv"])\
    .agg(count_drv = ("drv","count"))
    
# 168 page
# 사람은 5명인데 1명씩 다른 데이터 프레임 만들기
import pandas as pd
test1 = pd.DataFrame({
    "id": [1,2,3,4,5],
    "midterm" : [60,70,80,90,100]
})

test2 = pd.DataFrame({
    "id": [1,2,3,40,5],
    "final" : [70,83,65,95,80]
})

test1
test2

# 두 데이터 프레임 합치기
# on은 기준이 되는 것(left는 왼쪽것을 기준, light는 오른쪽거 기준, inner는 교집합만, outer는 합집합)
# left로 합치기
total = pd.merge(test1, test2, how="left", on="id")
total

# right로 합치기
total = pd.merge(test1, test2, how="right", on="id")
total

# inner로 합치기
total = pd.merge(test1, test2, how="inner", on="id")
total

# outer로 합치기
total = pd.merge(test1, test2, how="outer", on="id")
total

# 맞추는 거 없이 옆으로 다 합침
test_concat = pd.concat([test1, test2], axis=1)
test_concat

# 169 page
name = pd.DataFrame({
    "nclass" : [1,2,3,4,5],
    "teacher" : ["kim", "lee","park","choi","jung"]
})
name

# 외부데이터 불러와서 합치기
exam = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/exam.csv")
exam

pd.merge(exam,name, how="inner", on="nclass")

# 데이터 세로로 붙이기
score1 = pd.DataFrame({
    "id": [1,2,3,4,5],
    "score" : [60,70,80,90,100]
})

score2 = pd.DataFrame({
    "id": [6,7,8,9,10],
    "score" : [70,83,65,95,80]
})
score1
score2

score_all=pd.concat([score1, score2])
score_all


