import numpy as np
import pandas as pd

df = pd.DataFrame({"name":["김지훈","이유진","박동현","김민지"],
                   "english" : [90,80,60,70],
                   "math":[50,60,100,20]})
df

#
df["name"]
sum(df["english"])/4

x=pd.DataFrame({"제품":["사과","딸기","수박"],
"가격":[1800,1500,3000],
"판매량":[24,38,13]})
x
sum(x["가격"])/3
sum(x["판매량"])/3


# 86page
import pandas as pd

df_exam=pd.read_excel("data/excel_exam.xlsx")
df_exam
sum(df_exam["math"]) /20
sum(df_exam["english"])/20
sum(df_exam["science"])/20

len(df_exam)
df_exam.shape
df_exam.size

?pd.read_excel

df_exam2=pd.read_excel("data/excel_exam.xlsx",sheet_name="Sheet2")
df_exam2

df_exam["total"]  = df_exam["math"] + df_exam["english"] + df_exam["science"]
df_exam["total"]

df_exam["mean"]  = round((df_exam["math"] + df_exam["english"] + df_exam["science"])/3,2)
df_exam["mean"]



#판다스에서도 데이터프레임[데이터프레임수식] 하면 참인 것만 나옴(넘파이랑 같음)
df_exam[(df_exam["math"] > 50) & (df_exam["english"] > 50)]

df_exam[(df_exam["math"] > sum(df_exam["math"])/20 ) & (df_exam["english"] < sum(df_exam["english"])/20)]

# mean_m = np.mean(df_exam["math"])
# mean_e = np.mean(df_exam["english"])
# df_exam[(df_exam["math"] > mean_m ) & (df_exam["english"] < mean_e)]

# df_exam[df_exam["nclass"]==3][["math","english","science"]]
df_exam_nc3 = df_exam[df_exam["nclass"]==3]
df_exam_nc3[["math","english","science"]]
df_exam_nc3[0:2]

a = np.array([4,2,5,3,6])
a[2]

df_exam[0:10:2]
df_exam.sort_values("math")
df_exam.sort_values("math",ascending=False)
df_exam.sort_values(["nclass","math"],ascending=[True,False]) #반에서 오름차순 후, 반 내에서 수학 내림차순

np.where(a>3, "Up", "Down") #True면 UP, False 면 Down
df_exam["updown"]=np.where(df_exam["math"]>50, "Up", "down")
df_exam

# 복합적인 조건을 줘서 그 값 만족하면 a, 만족 안하면 b라는 값 줄 수 있음
