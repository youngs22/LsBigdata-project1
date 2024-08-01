# 균일분포
# X~U(a,b)
from scipy.stats import uniform

# 균일 분포에서 loc은 시작위치 scale은 끝나는 위치
uniform.rvs(loc=2, scale=4, size=1)
uniform.pdf(x, loc=2, scale=4)

# 시각화
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

k=np.linspace(0,8,100)
y=uniform.pdf(k,2,4)
plt.plot(k,y,color="black")

plt.show()
plt.clf()

# P(x<3.25)
uniform.cdf(3.25, loc=2, scale=4)
1.25 * (1/4)

# P(5<X<8.39)
uniform.cdf(8.39,2,4) - uniform.cdf(5,2,4)
uniform.cdf(8.39,2,4) - uniform.cdf(6,2,4) 
# 그래프가 6까지 이므로 6으로 해도 같음
1 * 0.25

# 상위 7% 값은?
uniform.ppf(0.93, 2, 4)

# 표본 20개를 뽑아서 표본 평균 계산
# rvs는 함수 밖에서 random.seed() 주거나, 함수안에서 random_state= 줌
uniform.rvs(loc=2, scale=4, size=20, random_state=42).mean()

x=uniform.rvs(loc=2, scale=4, size=20*1000, random_state=42)
x= x.reshape(-1,20)
blue_x = x.mean(axis=1)

# 그래프 그리기1----------------------------------------------
# 히스토그램
sns.histplot(blue_x, stat="density")
plt.show()

# X bar ~ N(mu, sigma^2/n) -> 모
# X bar ~ N(4, 1.33333/20) -> 표본
uniform.var(loc=2,scale=4) -> 모분산
uniform.expect(loc=2,scale=4) 

# 정규분포선
from scipy.stats import norm

xmin, xmax = (blue_x.min(), blue_x.max())
x_values = np.linspace(xmin, xmax, 100)
pdf_values = norm.pdf(x_values, loc=4, scale=np.sqrt(1.33333/20))
plt.plot(x_values, pdf_values, color="red", linewidth=2)

plt.show()
plt.clf()

# 정규분포선과 평균선 
from scipy.stats import norm

x_values = np.linspace(3, 5, 100)
pdf_values = norm.pdf(x_values, loc=4, scale=np.sqrt(1.33333/20))
plt.plot(x_values, pdf_values, color="red", linewidth=2)
plt.axvline(x=4.0, color='black', linestyle="--", linewidth=1, label='Mean')

# 표본 평균(파란벽돌) 점찍기
blue_x=uniform.rvs(2,4,20).mean()
plt.scatter(blue_x, 0.002, color="blue", zorder=10, s=10)

# 신뢰구간 선
# 유의수준 = 0.05
# norm.ppf(0.975, loc=0, scale=1) = 1.96
a=blue_x + 1.96 * np.sqrt(1.33333/20)
b=blue_x - 1.96 * np.sqrt(1.33333/20)
plt.axvline(x=a, color='green', linestyle="--", linewidth=1)
plt.axvline(x=b, color='green', linestyle="--", linewidth=1)
plt.show()
plt.clf() 

# X bar ~ N(4, 1.33333/20), 신뢰구간 95% 안에서 표본20개 추출
a = norm.ppf(0.025,loc=4,scale=np.sqrt(1.33333/20))
b = norm.ppf(0.975,loc=4,scale=np.sqrt(1.33333/20))
4-a
4-b

# X bar ~ N(4, 1.33333/20), 신뢰구간 99% 안에서 표본20개 추출
a = norm.ppf(0.005,loc=4,scale=np.sqrt(1.33333/20))
b = norm.ppf(0.995,loc=4,scale=np.sqrt(1.33333/20))
4-a
4-b

# 57page 신뢰구간 구하기 문제
from scipy.stats import norm
x = np.array([79.1,68.8,62.0,74.4,71.0,60.6,98.5,86.4,73.0,40.8,61.2,68.7,61.6,67.7,61.7,66.8])
x.mean()

# 방법1
a = norm.ppf(0.05,loc=68.89,scale=(6/4))
b = norm.ppf(0.95,loc=68.89,scale=(6/4))
68.89-a
68.89-b

# 방법2
z_005 = norm.ppf(0.95, loc=0, scale=1)
z_005
x.mean() + z_005 * 6 / np.sqrt(16)
x.mean() - z_005 * 6 / np.sqrt(16)


