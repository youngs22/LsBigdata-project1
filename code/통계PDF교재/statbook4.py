# 2x+3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib
import pandas as pd

x=np.linspace(0,100,400)
y= 2*x+3

np.random.seed(20240805)
obs_x=np.random.choice(np.arange(100),20)
epsilon_i=norm.rvs(loc=0,scale=100,size=20)
obs_y=2*obs_x+3+epsilon_i

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['axes.unicode_minus'] = False
# plt.plot(x,y,label="2x+3", color="black")
plt.scatter(obs_x,obs_y,color="blue", s=4)
plt.show()
plt.clf()
-----------------------------------------------------------------------------------
# 위에거랑 회귀식이랑 같이 그리기
# 위에거 그리기
x=np.linspace(0,100,400)
y= 2*x+3

# np.random.seed(20240805)
obs_x=np.random.choice(np.arange(100),20)
epsilon_i=norm.rvs(loc=0,scale=100,size=20)
obs_y=2*obs_x+3+epsilon_i

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['axes.unicode_minus'] = False
plt.plot(x,y,label="2x+3", color="black")
plt.scatter(obs_x,obs_y,color="blue", s=4)
plt.show()

# 선형 회귀식 그리기
from sklearn.linear_model import LinearRegression

obs_x=np.random.choice(np.arange(100),20)
obs_y=2*obs_x+3+epsilon_i

df=pd.DataFrame(
    {
        "x" : obs_x,
        "y" : obs_y
    }
)
model = LinearRegression()

obs_x=obs_x.reshape(-1,1)

model.fit(obs_x, obs_y) # 자동으로 기울기 및 절편 값 구해줌

x=np.linspace(0,100,400)
y=model.coef_[0] * x + model.intercept_
plt.plot(x,y,color="red")

plt.show()
plt.clf()

#
!pip install statsmodels
import statsmodels.api as sm

obs_x = sm.add_constant(obs_x)
model = sm.OLS(obs_y,obs_x).fit()
print(model.summary())

