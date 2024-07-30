import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 불러오기
raw_welfare = pd.read_spss("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/koweps/Koweps_hpwc14_2019_beta2.sav")
raw_welfare

# 복사본 만들기
welfare = raw_welfare.copy()
welfare
welfare.shape
welfare.info()
welfare.describe()

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

-----------------------------------------------------------------------
#9-2 성별에 따른 월급 차이

# welfare["sex"].dtypes
# welfare["sex"].value_counts()
# welfare["sex"].isna().sum()

welfare["sex"] = np.where(welfare["sex"] == 1.0, "male", "female")
welfare["sex"]

welfare["income"].dtypes
welfare["income"].describe()
welfare["income"].isna().sum()

# 무응답을 9999로 처리했다고 코드북에 나와있늠
# 만약 max가 9999로 나와있으면 결측치 처리 해줘야함
# 현재는 안나와있음
welfare["income"] = np.where(welfare["income"] == 9999, np.nan, welfare["income"])
welfare["income"].isna().sum()

# income의 결측치제외하고 성별로 평균 구하기
sex_income = welfare.dropna(subset="income")\
        .groupby("sex", as_index = False)\
        .agg(mean_income = ("income","mean"))
sex_income

# 그래프 그리기
sns.barplot(data=sex_income, x="sex", y="mean_income")
plt.show()
plt.clf()

----------------------------------------------------------------------------
# 숙제 : 위 그래프에서 각 성별 95% 신뢰구간 계산후 그리기, 위아래 검정색 막대기
sex_income = welfare.dropna(subset="income")\
        .groupby("sex", as_index = False)\
        .agg(mean_income = ("income","mean"),
             std_income = ("income","std"),
             count = ("sex","count"))
sex_income


# 각 성별의 95% 신뢰구간 계산
sex_income['ci_upper'] = sex_income['mean_income'] + 1.96 * sex_income['std_income'] / np.sqrt(sex_income['count'])
sex_income['ci_lower'] = sex_income['mean_income'] - 1.96 * sex_income['std_income'] / np.sqrt(sex_income['count'])

# 막대 그래프 그리기
sns.barplot(data=sex_income, x='sex', y='mean_income', hue="sex", errorbar=None)

# 신뢰구간을 나타내는 에러바 추가
plt.errorbar(
    x=sex_income['sex'],  # x축 위치
    y=sex_income['mean_income'],  # y축 평균 소득
    yerr=[sex_income['mean_income'] - sex_income['ci_lower'], sex_income['ci_upper'] - sex_income['mean_income']],  # 신뢰구간 범위
    fmt='none',  # 데이터 포인트는 표시하지 않음
    capsize=5,  # 에러바 끝에 캡의 크기
    color='black'  # 에러바 색상
)
plt.show()
plt.clf()

------------------------------------------------------------------------------
# 9-3 나이와 월급의 관계
# birth는 출생년도
welfare["birth"].describe()
sns.histplot(data=welfare, x="birth")
plt.show()
plt.clf()

# 나이 age 변수 만들기
welfare = welfare.assign(age=2019-welfare["birth"]+1)
welfare["age"].describe()
sns.histplot(data=welfare, x="age")
plt.show()
plt.clf()

# 그래프 그리기
age_income = welfare.dropna(subset="income")\
        .groupby("age", as_index = False)\
        .agg(mean_income = ("income","mean"))
age_income

sns.lineplot(data=age_income, x="age", y="mean_income")
plt.show()
plt.clf()

# 나이별 income 칼럼 na 개수 세기
my_df = welfare.assign(income_na = welfare["income"].isna())\
                    .groupby("age", as_index=False)\
                    .agg(n = ("income_na","count"))
                    
sns.barplot(data=my_df, x="age", y="n")
plt.xticks(rotation=90, fontsize=7)
plt.show()
plt.clf()
----------------------------------------------------------------------
# 9-4 연령대별 월급 차이
welfare = welfare.assign(ageg = np.where(welfare["age"] < 30, "young",
                                np.where(welfare["age"]<=59, "middle","old")))
welfare["ageg"].value_counts()

sns.countplot(data=welfare, x="ageg", hue="ageg")
plt.show()
plt.clf()

ageg_income = welfare.dropna(subset="income")\
        .groupby("ageg", as_index = False)\
        .agg(mean_income = ("income","mean"))
ageg_income

sns.barplot(data=ageg_income, x="ageg", y="mean_income", hue="ageg")
plt.show()
plt.clf()

## 막대 정렬하기
sns.barplot(data=ageg_income, x="ageg", y="mean_income", hue="ageg",
            order = ["young","middle","old"])
            
welfare = welfare.assign(ageg = np.where(welfare["age"] < 30, "young",
                                np.where(welfare["age"]<=59, "middle","old")))
welfare["ageg"].value_counts()

sns.countplot(data=welfare, x="ageg", hue="ageg")
plt.show()
plt.clf()

========================================================================
# 10살 별로 나이 나누는 방법
# 방법1 - 노가다
welfare = welfare.assign(ageg2 = np.where(welfare["age"]<10, "0years",
                                 np.where(welfare["age"]<20, "10years",
                                 np.where(welfare["age"]<30, "20years",
                                 np.where(welfare["age"]<40, "30years",
                                 np.where(welfare["age"]<50, "40years",
                                 np.where(welfare["age"]<60, "50years",
                                 np.where(welfare["age"]<70, "60years",
                                 np.where(welfare["age"]<80, "70years",
                                 np.where(welfare["age"]<90, "80years",
                                 np.where(welfare["age"]<100, "90years",
                                 np.where(welfare["age"]<110, "100years","110years",))))))))))))
welfare["ageg2"].value_counts()

ageg2_income = welfare.dropna(subset="income")\
        .groupby("ageg2", as_index = False)\
        .agg(mean_income = ("income","mean"))
ageg2_income

sns.barplot(data=ageg_income, x="ageg2", y="mean_income", hue="ageg")
plt.show()
plt.clf()

#방법2-cut활용
plt.rcParams['font.family'] ='Malgun Gothic'
bin_cut = np.array([0,9,19,29,39,49,59,69,79,89,99,109,119])

welfare = welfare.assign(age_group = pd.cut(welfare["age"],
                          bins = bin_cut,
                          labels = (np.arange(12)*10).astype(str) + "대"))
welfare["age_group"]

age_income = welfare.dropna(subset="income")\
                    .groupby("age_group", as_index = False)\
                    .agg(mean_income = ("income","mean"))
age_income

sns.barplot(data=age_income, x="age_group", y="mean_income")
plt.show()
plt.clf()
-----------------------------------------------------------------
# 9-5 연령대 및 성별 월급 차이
# 나이 세그룹과 성별간의 평균 수입 막대그래프
sex_income = welfare.dropna(subset="income")\
        .groupby(["ageg","sex"], as_index = False)\
        .agg(mean_income = ("income","mean"))
sex_income

sns.barplot(data=sex_income, x="ageg", y="mean_income", hue="sex",
            order = ["young","middle","old"])
plt.show()
plt.clf()

# 나이대와 성별간의 평균 수입 막대그래프
plt.rcParams['font.family'] ='Malgun Gothic'
bin_cut = np.array([0,9,19,29,39,49,59,69,79,89,99,109,119])

welfare = welfare.assign(age_group = pd.cut(welfare["age"],
                          bins = bin_cut,
                          labels = (np.arange(12)*10).astype(str) + "대"))
                          
welfare["age_group"] = welfare["age_group"].astype("object")

age_sex_income = welfare.dropna(subset="income")\
                    .groupby(["age_group","sex"], as_index = False)\
                    .agg(mean_income = ("income","mean"))
age_sex_income

sns.barplot(data=age_sex_income, x="age_group", y="mean_income",hue="sex")
plt.show()
plt.clf()

# 나이와 성별간의 선그래프
sex_income = welfare.dropna(subset="income")\
        .groupby(["age","sex"], as_index = False)\
        .agg(mean_income = ("income","mean"))
sex_income

sns.lineplot(data=sex_income, x="age", y="mean_income", hue="sex")
plt.show()
plt.clf()

#
sex_income = welfare.dropna(subset="income")\
        .groupby(["ageg","sex"], as_index = False)\
        .agg(mean_income = ("income","mean"))
sex_income

sns.barplot(data=sex_income, x="ageg", y="mean_income", hue="sex",
            order = ["young","middle","old"])
plt.show()
plt.clf()

-----------------------------------------------------------------------
# 사용자 정의 함수
def custom_mean(series, dropna = True):
    if dropna :
        return print(series, "hi")
    else :
        return print(series, "hello")
    
#그룹화 및 사용자 정의 함수 적용
sex_age_income = welfare.dropna(subset = ["age_group", "sex"]) \
                        .groupby(["age_group", "sex"], as_index = False) \
                        .agg(mean_income = ("income", "custom_mean")) # 오류남
                        
sex_age_income = welfare.dropna(subset = ["age_group", "sex"]) \
                        .groupby(["age_group", "sex"], as_index = False) \
                        .agg(mean_income = ("income", lambda x: custom_mean(x, dropna=False)))
x = np.arange(10)
np.quantile(x, q=0.5)

print(sex_age_income) 


# 연령대별 성별 상위 4% 수입
plt.rcParams['font.family'] ='Malgun Gothic'
bin_cut = np.array([0,9,19,29,39,49,59,69,79,89,99,109,119])

welfare = welfare.assign(age_group = pd.cut(welfare["age"],
                          bins = bin_cut,
                          labels = (np.arange(12)*10).astype(str) + "대"))
                          
welfare["age_group"] = welfare["age_group"].astype("object")

age_sex_income = welfare.dropna(subset="income")\
                    .groupby(["age_group","sex"], as_index = False)\
                    .agg(top4per_income = ("income", lambda x : np.quantile(x, q=0.96)))
age_sex_income

