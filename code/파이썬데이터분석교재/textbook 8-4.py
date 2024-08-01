import pandas as pd
import seaborn as sns

economics = pd.read_csv("data/economics.csv")
economics

sns.lineplot(data=economics, x="date", y="unemploy")

plt.show()
plt.clf()

# 날짜 시간 타입 변수 만들기
economics["date2"] = pd.to_datetime(economics["date"])
economics.info()

economics[["date", "date2"]]
economics["date2"].dt.year # 어트리뷰트이기 때문에 괄호 안붙임
economics["date2"].dt.month
economics["date2"].dt.day
economics["date2"].dt.month_name()
economics["date2"].dt.quarter
economics["quarter"] = economics["date2"].dt.quarter
economics[["date2","quarter"]]

# 각 날짜는 무슨 요일인가
economics["date2"].dt.day_name()
economics["date2"] + pd.DateOffset(days=3)
economics["date2"] + pd.DateOffset(months=1)
economics["date2"] + pd.DateOffset(days=30)

# 연도 변수 만들기
economics["year"] = economics["date2"].dt.year
economics.info()

# x축에 연도 표시하기 & 신뢰구간 제거하기
sns.lineplot(data=economics, x="year", y="unemploy", errorbar=None)
sns.scatterplot(data=economics, x="year", y="unemploy",size=1)
plt.show()
plt.clf()

economics.head(10)

# 각 년도 표본 평균과 표준편차 구하기
my_df = economics.groupby("year", as_index = False) \
         .agg(
             mean_year = ("unemploy","mean"),
             std_year = ("unemploy","std"),
             mon_n = ("unemploy", "count")
         )

my_df.head(10)

my_df["mean_year"] + 1.96 * my_df["std_year"]/my_df["mon_n"] # 95%일때 신뢰구간 1.96

my_df["right_ci"] = my_df["mean_year"] + 1.96 * my_df["std_year"]/np.sqrt(my_df["mon_n"])
my_df["left_ci"] = my_df["mean_year"] - 1.96 * my_df["std_year"]/np.sqrt(my_df["mon_n"])
my_df["right_ci"]
my_df["left_ci"]

x = my_df["year"]
y = my_df["mean_year"]
plt.plot(x,y,color="black")
plt.scatter(x, my_df["left_ci"], color = "blue", s=2)
plt.scatter(x, my_df["right_ci"], color = "Green", s=2)
plt.show()
plt.clf()

