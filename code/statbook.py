from scipy.stats import bernoulli

# 베르누이 분포
#  bernoulli.pmf(k, p)
# 확률질량함수 pmf 확률변수가 갖는 값에 해당하 는 확률을 저장하고 있는 함수
# 베르누이 확률 변수의 기댓값 E(X)=p

# P(X=1)
 bernoulli.pmf(1,0.3)

# P(X=0)
 bernoulli.pmf(0,0.3)

# p가 0.3인 베르누이 분포에서 나올 수 있는 X확률변수
# rvs() 함수
bernoulli.rvs(0.3)

=========================================================================

# 이항분포 P(X=k|n,p)  
from scipy.stats import binom
# n:  베르누이 확률변수 더한 갯수
# p: 1이 나올 확률
# binom.pmf(k, n, p)
binom.pmf(0, n=2, p=0.3)
binom.pmf(1, n=2, p=0.3)
binom.pmf(2, n=2, p=0.3)

# X~B(n,p)
binom.pmf(0, n=30, p=0.3)
binom.pmf(1, n=30, p=0.3)
binom.pmf(2, n=30, p=0.3)

result = [binom.pmf(x, n=30, p=0.3) for x in range(31)]
result

# (54 26)팩토리알 ======================================================
import math
math.factorial(54) / (math.factorial(26) * math.factorial(28))
math.comb(54,26)

import numpy as np
fac_54=np.cumprod(np.arange(1,55))[-1] 
fac_26=np.cumprod(np.arange(1,27))[-1]
fac_28=np.cumprod(np.np.arange(1,29))[-1]

# math.log(math.factorial(54)) - (math.log(math.factorial(26)+math.factorial(28)))

logf_54 = sum(np.log(np.arange(1,55)))
logf_26 = sum(np.log(np.arange(1,27)))
logf_28 = sum(np.log(np.arange(1,29)))
np.exp(logf_54-(logf_26 + logf_28))
=========================================================================

# 이항분포 이어서,,,
math.comb(2,0) * 0.3**0 * (1-0.3)**2
math.comb(2,1) * 0.3**1 * (1-0.3)**1
math.comb(2,2) * 0.3**2 * (1-0.3)**0

# 위에 일일이 한거 아래처럼 이항분포 확률질량함수로 씀
# pmf : probability 
binom.pmf(0, 2, p=0.3)
binom.pmf(1, 2, p=0.3)
binom.pmf(2, 2, p=0.3)

# Q1. P(X=4)
# X ~ B(n=10, p=0.36)
binom.pmf(4, 10, p=0.36)

# Q2. P(X<=4)
# X ~ B(n=10, p=0.36)
binom.pmf(np.arange(5), 10, p=0.36).sum()

# Q2. P(2<X<=8)
# X ~ B(n=10, p=0.36)
binom.pmf(np.array([3,4,5,6,7,8]), 10, p=0.36).sum()
binom.pmf(np.arange(3,9), 10, p=0.36).sum()

# X~B(30,0.2)
# 확률변수 X가 4보다 작거나 25보다 크거나 같을 확률을 구하시오

#1번 방법
1- binom.pmf(np.arange(4,25), n=30, p=0.2).sum()

#2번 방법
a=binom.pmf(np.arange(4), n=30, p=0.2).sum() 
b=binom.pmf(np.arange(25,31), n=30, p=0.2).sum()
a+b

# p가 0.3인 이항 분포에서 나올 수 있는 X확률변수
# rvs() 함수
bernoulli.rvs(0.3) + bernoulli.rvs(0.3)
binom.rvs(2,0.3)

# X~B(30,0.26)
# 표본 30개 뽑기
binom.rvs(30,0.26,size=30)

# 이항 분포 확률 변수의 기댓값 E(X)= n*p
30 * 0.26

#시각화
import matplotlib.pyplot as plt
import seaborn as sns

x=np.arange(31)
x_1=binom.rvs(n=30, p=0.26, size=10)
mean = np.mean(binom.pmf(x, n=30, p=0.26))

prob_x=binom.pmf(x, n=30, p=0.26)
sns.barplot(prob_x, color="blue")

# 구슬표시하기
plt.scatter(x_1, np.repeat(0.002,10), color="red", zorder=100, s=10)

# 평균표시하기
plt.axvline(x=7.8, color='red', linestyle="--", linewidth=1, label='Mean')
plt.xticks(fontsize=6)
plt.show()
plt.clf()

# plt.clf()
# x=binom.rvs(30,0.26,size=30)
# plt.hist(x, bins=np.arange(min(x), max(x) + 1), edgecolor='black', alpha=0.7)
# plt.show()

# 교재 p.207
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.clf()
x= np.arange(31)
df = pd.DataFrame({"x": x, "prob":prob_x})
sns.barplot(data=df, x="x", y="prob")
plt.xticks(fontsize=6)
plt.show()


# 누적확률분포(cdf)
# P(4<X<=18) 
binom.cdf(18, n=30, p=0.26) - binom.cdf(4, n=30, p=0.26)

# P(13<X<20)
binom.cdf(19, n=30, p=0.26) - binom.cdf(13, n=30, p=0.26)

#
binom.ppf(0.5, n=30, p=0.26)
binom.cdf(8, n=30, p=0.26)
binom.cdf(7, n=30, p=0.26)

# 
binom.ppf(0.7, n=30, p=0.26)
binom.cdf(9, n=30, p=0.26)
binom.cdf(8, n=30, p=0.26)

=========================================================================

# 정규분포
# mu = 0, 시그마 = 1
1/np.sqrt(2 *math.pi)
from scipy.stats import norm
norm.pdf(0, loc=0, scale=1)

# mu = 3, 시그마 = 4
norm.pdf(5, loc=3, scale=4)

k = np.linspace(-3, 3, 100)
y = norm.pdf(k, loc=0, scale=1)
# plt.scatter(k, y, color="red", s=1)
plt.plot(k, y, color="black")
plt.show()
plt.clf()

# loc 조정시 분포 왼쪽 오른쪽으로 움직임
k = np.linspace(-5, 5, 100)

y = norm.pdf(k, loc=0, scale=1)
plt.plot(k, y, color="blue")

y = norm.pdf(k, loc=3, scale=1)
plt.plot(k, y, color="black")

y = norm.pdf(k, loc=-3, scale=1)
plt.plot(k, y, color="black")

plt.show()
plt.clf()

# sigma(scale) : 분포의 퍼짐을 결정하는 모수(표준편차)
# scale이 작을수록 분포 뾰족해짐
k = np.linspace(-5, 5, 100)

y = norm.pdf(k, loc=0, scale=1)
plt.plot(k, y, color="blue")

y = norm.pdf(k, loc=0, scale=0.5)
plt.plot(k, y, color="black")

y = norm.pdf(k, loc=0, scale=2)
plt.plot(k, y, color="green")

plt.show()
plt.clf()

# 100까지 누적분포함수 1임
norm.cdf(100, loc=0, scale=1)

# 정규분포에서는 등호 신경 x
norm.cdf(0.54, loc=0, scale=1) - norm.cdf(-2, loc=0, scale=1) 

# Q1.
norm.cdf(1, loc=0, scale=1) + (1-norm.cdf(3, loc=0, scale=1))
norm.cdf(1, loc=0, scale=1) + norm.cdf(-3, loc=0, scale=1)

# Q2. P(3<X<5)=?  -> 15.54%
norm.cdf(5, loc=3, scale=5) - norm.cdf(3, loc=3, scale=5) 

# 위 확률변수에서 표본 1000개 뽑기
x = norm.rvs(loc=3, scale=5, size=1000)
sum((x > 3) & (x < 5)) / 1000
# x[np.where((x > 3) & (x < 5))]

# 평균 0, 표준편차 1
# 표본 1000개 뽑아서 0보다 작은 비율 확인
x = norm.rvs(loc=0, scale=1, size=1000)
sum(x < 0) / 1000
np.mean(x<0)
# x[np.where(x<0)]

# 그래프 그리기
x = norm.rvs(loc=3, scale=2, size=1000)
sns.histplot(x)
sns.histplot(x, stat="density")

xmin, xmax = (x.min(), x.max())
x_values = np.linspace(xmin, xmax, 100)
pdf_values = norm.pdf(x_values, loc=3, scale=2)
plt.plot(x_values, pdf_values, color="red", linewidth=1)

plt.show()
plt.clf()


# LS빅데이터 스쿨 학생들의 중간고사 점수는 
# 평균이 30이고, 분산이 4인 정규분포
# 상위 5%에 해당하는 학생 점수는?
import numpy as np
from scipy.stats import norm

x=np.arange(28)
k = norm.ppf(0.95, loc=30, scale=2)
k



