# p 129
!pip install palmerpenguins
import pandas as pd
import numpy as np
import plotly.express as px
from palmerpenguins import load_penguins

penguins = load_penguins()

penguins.head()
penguins.info()
penguins["species"].unique()
penguins.columns

fig = px.scatter(
    penguins,
    x= "bill_length_mm",
    y= "bill_depth_mm",
    color="species",
  # trendline="ols", # 각각 추세선 그려줌
    size_max=15
    )
    
# 레이아웃 업데이트
fig.update_layout(
    title=dict(text="팔머펭귄 종별 부리 길이 vs. 깊이", font=dict(color="white", size=24)),
    paper_bgcolor="black",
    plot_bgcolor="black",
    font=dict(color="white"),
    xaxis=dict(
        title=dict(text="부리 길이 (mm)", font=dict(color="white")), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    yaxis=dict(
        title=dict(text="부리 깊이 (mm)", font=dict(color="white")), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    legend=dict(title=dict(text="펭귄 종"), font=dict(color="white"))
)
fig.update_traces(marker=dict(size=10,opacity=0.6))

fig.show()
----------------------------------------

from sklearn.linear_model import LinearRegression

model = LinearRegression()
penguins = penguins.dropna()
x=penguins[["bill_length_mm"]]
y=penguins["bill_depth_mm"]

model.fit(x,y)
linear_fit=model.predict(x)

fig.add_trace(
    go.Scatter(
        mode = "lines",
        x=penguins["bill_length_mm"], y=linear_fit,
        name="선형회귀선",
        line=dict(dash="dot",color="white")
    )
)

fig.show()

model.coef_      # 기울기, 부리길이가 1mm증가할때마다 부리깊이는 0.08mm 감소
model.intercept_ # 절편

# 전체로 회귀직선 구한거랑 종별로 따로 구한거랑 회귀직선 다름 -> Simpson's paradox
------------------------------------------
# 범주형 변수로 회귀분석 진행하기
# 범주형 변수인 'species'를 더미 변수로 변환
penguins_dummies = pd.get_dummies(penguins, 
                                  columns=['species'],
                                  drop_first=False)
penguins_dummies.columns
penguins_dummies.iloc[:,-3:]

# x와 y 설정
x = penguins_dummies[["bill_length_mm", "species_Chinstrap", "species_Gentoo"]] # x에 문자형이 아닌 숫자형으로 다 들어감
y = penguins_dummies["bill_depth_mm"]

# 모델 학습
model = LinearRegression()
model.fit(x, y) # 문자형 변수를 숫자형으로 더미처리

model.coef_      
model.intercept_

regline_y = model.predict(x)
# y= 0.20*bill_length_mm -1.93*species_Chinstrap -5.10*species_Gentoo + 10.56

# species    island  bill_length_mm  ...  body_mass_g     sex  year
# Adelie     Torgersen            39.5  ...       3800.0  female  2007
# Chinstrap  Torgersen            40.5  ...       3800.0  female  2007
# Gentoo     Torgersen            40.5  ...       3800.0  female  2007
# x1, x2, x3
# 39.5, 0, 0
# 40.5, 1, 0
# y = 0.2 * bill_length -1.93 * species_Chinstrap -5.1 * species_Gentoo + 10.56

import matplotlib.pyplot as plt
import seaborn as sns

sns.lineplot(x=x["bill_length_mm"], y=regline_y, s=1, hue=penguins["species"], pallete="deep")
sns.scatterplot(x["bill_length_mm"], y, color="black")
plt.show()
plt.clf()
