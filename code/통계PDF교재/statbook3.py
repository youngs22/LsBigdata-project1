# E[X^2] 구하기
# X ~ N(3,5^2)
x=norm.rvs(loc=3, scale=5, size=100000)
#1
np.mean(x**2)

#2
sum(x**2) / (len(x)-1)

#3
# 25 = E[x^2] - 3^2
# E[x^2] = 25+9 = 34

# 위에서 알 수 있는것은 공식대로 E[]안구하고 np.mean()해도 값 비슷하게 나옴
np.mean((x-x**2)/(2*x))
# -> 몬테카를로 적분 : 확률변수 기댓값을 구할때, 표본을 많이 뽑은 후 원하는 형태로
# 변형 및 평균을 계산해서 기대값을 구하는 방법

-----------------------------------------------------------
# 표본 10만개 뽑라서 s^2 구하기
np.random.seed(20240729)
x=norm.rvs(loc=3, scale=5, size=100000)

#1
x_bar = x.mean()
s_2 = sum((x-x_bar)**2)/ (100000-1)

#2
np.mean((x-x_bar)**2)

#3
np.mean(x**2) - (np.mean(x))**2

#
np.var(x, ddof=1) # n-1로 나눈 값
np.var(x, ddof=0) # n으로 나눈 값
np.var(x) # n으로 나눈 값임

# n-1 vs n
np.random.seed(20240729)
x=norm.rvs(loc=3, scale=5, size=10)
np.var(x)
np.var(x, ddof=1)   
                                                                            
--------------------------------------------------------------
# X~N(3,7^2)
from scipy.stats import norm
x= norm.ppf(0.25, loc=3, scale=7)
z= norm.ppf(0.25, loc=0, scale=1)
x
3+z*7 # x랑 같음

norm.cdf(5, loc=3, scale=7)
norm.cdf(2/7, loc=0, scale=1)

norm.ppf(0.975, loc=0, scale=1) # 양측 95% -> 1.96

# 표본 정규분포 표본 1000개 뽑아 히스토그램&정규분포선 그리기
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

z=norm.rvs(loc=0, scale=1, size=1000)
x=norm.rvs(loc=3, scale=2, size=1000)


sns.histplot(z, stat="density", color="gray")
sns.histplot(x, stat="density", color="skyblue")

total_min, total_max = (z.min(), x.max())

total_values = np.linspace(total_min, total_max, 500)

pdf_values_z = norm.pdf(total_values, loc=0, scale=1)
pdf_values_x = norm.pdf(total_values, loc=3, scale=2)
plt.plot(total_values, pdf_values_z, color="red", linewidth=2)
plt.plot(total_values, pdf_values_x, color="blue", linewidth=2)

plt.show()
plt.clf()

# 표준화
x=norm.rvs(loc=5, scale=3, size=1000)
z=(x-5)/3
sns.histplot(z, stat="density", color="gray")
z_min, z_max = (z.min(), z.max())

z_values = np.linspace(z_min, z_max, 500)

pdf_values = norm.pdf(z_values, loc=0, scale=1)
plt.plot(z_values, pdf_values, color="red", linewidth=2)

plt.show()
plt.clf()

# 
x=norm.rvs(loc=5, scale=3, size=20)
s=np.std(x, ddof=1)
s

x=norm.rvs(loc=5, scale=3, size=1000)
z=(x-5)/s
sns.histplot(z, stat="density", color="blue")

z_min, z_max = (z.min(), z.max())
z_values = np.linspace(z_min, z_max, 500)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
plt.plot(z_values, pdf_values, color="red", linewidth=2)

plt.show()
plt.clf()
----------------------------------------------------------------
# t분포에 대해서 알아보자
# x~t(df)
# 종모양, 대칭분포, 중심0
# 모수 df: 자유도 -> 분산의 영향 미침(퍼짐을 나타내는 모수)
# df가 작으면 분산 커짐

from scipy.stats import t
# t.pdf
# t.ppf
# t.cdf
# t.rvs
# 자유도가 4인 t분포의 pdf그리기
t_values = np.linspace(-4,4,100)
pdf_values = t.pdf(t_values, df=4)
plt.plot(t_values, pdf_values, color="red", linewidth=2)
plt.show()
plt.clf()

# t분포와 정규분포 비교하기
t_values = np.linspace(-4,4,100)
pdf_values_t = t.pdf(t_values, df=4)
plt.plot(t_values, pdf_values_t, color="black", linewidth=2)

pdf_values_z = norm.pdf(t_values, loc=0, scale=1)
plt.plot(t_values, pdf_values_z, color="red", linewidth=2)

plt.show()
plt.clf()

#
t_values = np.linspace(-4,4,100)
pdf_values = t.pdf(t_values, df=4)
plt.plot(t_values, pdf_values, color="red", linewidth=2)
plt.show()
plt.clf()

x=norm.rvs(loc=5, scale=3, size=20)
s=np.std(x, ddof=1)
s

x=norm.rvs(loc=5, scale=3, size=1000)
z=(x-5)/s
sns.histplot(z, stat="density", color="blue")

t_values = np.linspace(-4,4,100)
pdf_values = t.pdf(t_values, df=4)
plt.plot(t_values, pdf_values, color="red", linewidth=2)
plt.show()
plt.clf()

# X ~ ?(mu,sigma^2)
# X bar ~ N(mu, sigma^2/n)
# X bar ~= t(x_bar, s^2/n) 자유도:n-1
x=norm.rvs(loc=15,scale=3,size=16,random_state=42)
x
n=len(x)
x_bar=x.mean()

# 모평균에 대한 95% 신뢰구간 구하기
# 모분산을 모를 때
x_bar + t.ppf(0.975, df=n-1) * np.std(x,ddof=1) / np.sqrt(n)
x_bar - t.ppf(0.975, df=n-1) * np.std(x,ddof=1) / np.sqrt(n)

#모분산(3^2)을 알때
x_bar + norm.ppf(0.975, loc=0, scale=1) * 3 / np.sqrt(n)
x_bar - norm.ppf(0.975, loc=0, scale=1) * 3 / np.sqrt(n)

