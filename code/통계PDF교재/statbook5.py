import numpy as np
import pandas as pd

tab3 = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/tab3.csv")
tab3

tab1=pd.DataFrame({
    "id" : np.arange(1,13),
    "score" : tab3["score"]
})

tab2=tab1.assign(gender=["female"]*7 + ["male"]*5)

# 1표본 t테스트
# 귀무가설 vs 대립가설
# H0: mu=10 vs H1: mu!=10
# 유의수준 5%로 설정
from scipy.stats import ttest_1samp
result = ttest_1samp(tab1["score"], popmean=10, alternative='two-sided')
t_value = result[0] #검정통계량
p_value = result[1] #유의확률

tab1["score"].mean() #표본평균
result.pvalue
result.statistic
result.df
result.confidence_interval(confidence_level=0.95)
#신뢰구간
ci=result.confidence_interval(confidence_level=0.95)
ci[0]
ci[1]
# 귀무가설이 참일 때, 11.53이 관찰될 확률이 6.84%이므로, 
# 이것은 우리가 생각하는 보기 힘들다고 판단하는 기준인 유의수준(0.05)보다 크므로
# 귀무가설이 거짓이라고 보기 힘들다
# 유의확률 0.0648이 유의수준보다 크기때문에 귀무가설을 기각하지 못한다.
-----------------------------------

# 2표본 t검정
# 유의수준 1%, 두 그룹 분산은 같다고 설정
# H0: mu_m = mu_f  vs  H1: mu_m > mu_f
from scipy.stats import ttest_ind

male = tab2[tab2['gender'] == 'male']
female = tab2[tab2['gender'] == 'female']
result = ttest_ind(female['score'], male['score'], alternative='less',equal_var=True)
# alternative는 대립가설 기준으로 앞변수(female score)이 뒤보다 작다는 주장이니 less
result.statistic
result.pvalue
ci=result.confidence_interval(0.95)
ci[0]
ci[1]
---------------------

# 대응표본 t검정(짝지을 수 있는 표본)
# H0: mu_d = 0 va H1: mu_d > 0
# mu_d = mu_after - mu_before
tab3=tab3.pivot_table(
    index="id",
    columns="group",
    values="score"
).reset_index()
tab3

tab3_data['score_diff'] = tab3_data['after'] - tab3_data['before']
test3_data = tab3_data[['score_diff']]
test3_data

result = ttest_1samp(test3_data['score_diff'], popmean=0, alternative='two-sided')
result.statistic
result.pvalue
------------------------------------

# 연습1
# pivot&melt: long to wide, wide to long 
df = pd.DataFrame({
    "id": [1,2,3],
    "A": [10,20,30],
    "B": [40,50,60]
})
df_long = df.melt(id_vars="id",
        value_vars=["A","B"],
        var_name="group",
        value_name="score"
)
df_long.pivot_table(
    columns="group",
    values="score",
    aggfunc="mean"
)

# 연습2
import seaborn as sns
tips = sns.load_dataset("tips")
tips = tips.reset_index(drop=False)

tips_1=tips.pivot_table(
    index="index",
    columns="day",
    values="tip").reset_index()
