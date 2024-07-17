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
