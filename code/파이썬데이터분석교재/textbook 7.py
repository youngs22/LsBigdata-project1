import pandas as pd
import numpy as np
#178 page

df = pd.DataFrame({
    "sex":["M","F",np.nan,"M","F"],
    "score":[5,4,3,4,np.nan]
})
df

# 결측치 확인하기, isna() : 빈것만 True도 표시
pd.isna(df).sum()

# 결측치 제거하기
df.dropna() # 모든 변수 결측치 제거
df.dropna(subset="score") #특정 변수 결측치 제거
df.dropna(subset=["score","sex"]) #여러 변수 결측치 제거

# 데이터프레임 location을 사용한 인덱싱
# exam.loc[행 인덱스, 열 인덱스]
exam.iloc[0:2,0:4]
exam.loc[[2,7,4],["math"]]=np.nan
# exam.iloc[[2,7,4],2]=np.nan
 exam

df
df.loc[df["score"]==3.0, ["sex"]] = "M" 
df

# 수학 점수 50점 이하인 학생들 점수 50점으로 상향 조정
exam = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/exam.csv")
exam
exam.loc[exam["math"]<=50, "math"] = 50
exam

# 영어 점수 90 점 이상을 90점으로 하향 조정
exam.iloc[exam["english"]>=90, 3] = 90
exam
exam.iloc[exam[3:0]>=90, "english"] = 90
exam

exam.iloc[exam["english"]>=90, 3] #실행 안됨

exam.iloc[exam[exam["english"]>=90].index, 3] # index 됨
exam.iloc[np.where(exam["english"]>=90)[0], 3] # 튜플이라 됨
exam.iloc[np.array(exam["english"]>=90), 3] # 실행 됨

# math 50점 이하 -로 변경
exam.loc[exam["math"]<=50, "math"] = "-"
exam # 경고가 뜨지만 되긴 함

# "-" 결측치를 수학점수 평균으로 바꾸기
exam = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/exam.csv")
exam.loc[exam["math"]<=50, "math"] = "-"

#1
exam.loc[exam["math"] == "-", "math"] = np.nan
exam["math"] = exam["math"].fillna(exam["math"].mean())
exam

#2
math_mean = exam.loc[exam["math"] != "-", "math"].mean()
exam.loc[exam["math"] == "-", "math"] = math_mean
exam

#3
math_mean2 = exam.query('math not in ["-"]')["math"].mean()
exam.loc[exam["math"] == "-", "math"] = math_mean2
exam

#4
math_mean3 = exam[exam["math"] != "-"]["math"].mean()
exam.loc[exam["math"] == "-", "math"] = math_mean3
exam

#5
vector1 = np.array([np.nan if x =="-" else float(x) for x in exam["math"]])
np.nanmean(np.array([np.nan if x =="-" else float(x) for x in exam["math"]]))

vector2 = np.array([float(x) if x !="-" else np.nan for x in exam["math"]])
np.nanmean(np.array([float(x) if x !="-" else np.nan for x in exam["math"]]))

# 6
math_mean = exam.loc[exam["math"] != "-", "math"].mean()
exam["math"] = exam["math"].replace("-",math_mean)
exam

