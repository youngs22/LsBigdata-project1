import pandas as pd
import numpy as np

exam = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/exam.csv")
exam
exam.head(10) 
exam.tail(10)
exam.shape #튜플로 되어있음
exam.info()
exam.describe()

# 메서드 vs 속성(어트리뷰트)
# 메서드는 괄호있음, 속성은 괄호 없음

#csv 파일 불러와서 판다스 데이터프레임으로 만듬
type(exam)
var=[1,2,3]
type(var) #리스트임
exam.head()
var.head() # head 메소드가 list에는 없음
  
exam2 = exam.copy()
exam2.rename(columns={"nclass":"class"}) # 임시적으로 바꾼거라 exam2에 넣어줘야됨
exam2
exam2 = exam2.rename(columns={"nclass":"class"})

exam2["total"]=exam["math"] + exam["english"] +exam["science"]
exam2
exam2["pass_fail"] = np.where(exam2["total"]>=200, "pass" , "fail")
exam2["pass_fail"].value_counts()


import matplotlib.pyplot as plt
exam_count=exam2["pass_fail"].value_counts(rot=0) # rot=0으로 축이름 수평으로 만들기
plt.clf()
exam_count.plot.bar()
plt.show()

exam2["grade"]=np.where(exam2["total"]>=200, "A",
                np.where(exam2["total"]>=100, "B","C"))

exam2

exam2["grade"].isin(["A","C"]) #isin() 괄호안 조건 만족하는지 확인해줌
exam2["grade"].value_counts()

# 랜덤하게 수 뽑기
a = np.random.choice(np.arange(1,5),100,True,np.array([4/10,3/10,2/10,1/10]))
a
