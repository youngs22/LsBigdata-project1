import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyreadstat
# 9-6
# 데이터 불러오기
raw_welfare = pd.read_spss("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/koweps/Koweps_hpwc14_2019_beta2.sav")
raw_welfare

# 복사본 만들기
welfare = raw_welfare.copy()

# 변수명 바꾸기
welfare= welfare.rename(
    columns = {
        "h14_g3" : "sex",
        "h14_g4" : "birth",
        "h14_g10" : "marrige_type",
        "h14_g11" : "religion",
        "p1402_8aq1" : "income",
        "h14_eco9" : "code_job",
        "h14_reg7" : "code_region"
    }
)
welfare.info()
welfare = welfare[["sex","birth", "marrige_type", "religion","income","code_job","code_region"]]
welfare

welfare["sex"] = np.where(welfare["sex"] == 1.0, "male", "female")
welfare["sex"]

welfare = welfare.assign(age=2019-welfare["birth"]+1)
welfare["age"].describe()

welfare["code_job"].dtypes
welfare["code_job"].value_counts()

list_job = pd.read_excel("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/Koweps_Codebook_2019.xlsx",sheet_name="직종코드")
list_job

# welfare에 list_job 결합하기
welfare = welfare.merge(list_job, how="left", on="code_job")

# code_job의 결측치 제거하고 code_job과 job만 출력
welfare.dropna(subset="code_job")[["code_job","job"]]


# 직업별 월급 평균표 만들기
job_income = welfare.dropna(subset=["job","income"])\
                    .groupby("job", as_index = False)\
                    .agg(mean_income = ("income","mean"))
job_income.head()                    

# 그래프 그리기
top10 = job_income.sort_values("mean_income", ascending = False).head(10)
top10

plt.rcParams['font.family'] ='Malgun Gothic'

sns.barplot(data=top10, y="job", x="mean_income", hue="job")
plt.tight_layout() # 그래프 레이아웃 자동으로 조절
plt.show()
plt.clf()

# 9-7
# query 사용해 여자만 계산
df = welfare.dropna(subset=["job","income"])\
                    .query("sex == 'female'")\
                    .groupby("job", as_index = False)\
                    .agg(mean_income = ("income","mean"))\
                    .sort_values("mean_income", ascending = False)\
                    .head(10)

sns.barplot(data=df, y="job", x="mean_income", hue="job")
plt.tight_layout() # 그래프 레이아웃 자동으로 조절
plt.show()
plt.clf()

# query 사용해 남자만 계산
df = welfare.dropna(subset=["job","income"])\
                    .query("sex == 'male'")\
                    .groupby("job", as_index = False)\
                    .agg(mean_income = ("income","mean"))\
                    .sort_values("mean_income", ascending = False)\
                    .head(10)

sns.barplot(data=df, y="job", x="mean_income", hue="job")
plt.tight_layout() # 그래프 레이아웃 자동으로 조절
plt.show()
plt.clf()

# 9-8
welfare.info()
welfare["marrige_type"]
df = welfare.query("marrige_type != 5")\
            .groupby("religion", as_index = False)\
            ["marrige_type"]\
            .value_counts(normalize=True)
df

df.query("marrige_type == 1")\
    .assign(proportion=df["proportion"]**100)\
    .round(1)
    
================================================================    
