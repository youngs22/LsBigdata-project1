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
                                                                            
