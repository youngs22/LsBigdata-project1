import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 직선그리기
a=4
b=0

# x=np.arange(100)
x=np.linspace(-5,5,100)
y= a*x + b

plt.plot(x,y,color="blue")
plt.axvline(0, color="black")
plt.axhline(0, color="black")
plt.show()
plt.xlim(-5,5)
plt.ylim(-10,10)

plt.clf()

---------------------------------------------------------------
# 임의의 회귀식과 데이터 산점도
import pandas as pd
import numpy as np


a=80
b=0

x=np.linspace(-5,5,100)
y= a*x + b

house_train = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/train.csv")
house_train.shape
house_train.info()

my_df=house_train[["BedroomAbvGr", "SalePrice"]].head(10)
my_df["SalePrice"]=my_df["SalePrice"]/1000
plt.scatter(x=my_df["BedroomAbvGr"], y=my_df["SalePrice"])
plt.plot(x,y,color="black")
plt.xlim(0)
plt.lim(0)
plt.show()
plt.clf()

import numpy as np

house_train = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/train.csv")
house_train.shape
house_train.info()

my_df=house_train[["BedroomAbvGr", "SalePrice"]].head(10)
my_df["SalePrice"]=my_df["SalePrice"]/1000
plt.scatter(x=my_df["BedroomAbvGr"], y=my_df["SalePrice"])
plt.plot(x,y,color="black")
plt.xlim(0)
plt.lim(0)
plt.show()
plt.clf()

------------------------
# house 데이터로 회귀식 만들기
import numpy as np
import pandas as pd

# 예제 데이터
x = house_train["BedroomAbvGr"]
y = house_train["SalePrice"]/1000

# 평균 계산
x_mean = np.mean(x)
y_mean = np.mean(y)

# Sxx 계산
Sxy = np.sum((x - x_mean) * (y - y_mean))

# Sxy 계산
Sxx = np.sum((x - x_mean)**2)

# 결과 출력
print(f"Sxx = {Sxx}")
print(f"Sxy = {Sxy}")

# 회귀 계수 계산
beta_1 = Sxy / Sxx
beta_0 = y_mean - (beta_1 * x_mean)

beta_1
beta_0


a = beta_1
b = beta_0
x = np.linspace(0, 5, 100)
y = a * x + b

house_train = pd.read_csv("./data/houseprice/train.csv")
my_df = house_train[["BedroomAbvGr", "SalePrice"]].head(10)
my_df["SalePrice"] = my_df["SalePrice"] / 1000
plt.scatter(x = my_df["BedroomAbvGr"], y = my_df["SalePrice"])
plt.plot(x, y, color = "red")
plt.show()
plt.clf()

----------------------------------------------------------------
# 값 예측한거 파일 추출
#test 파일에서 id랑년도만 빼줌
house_test = pd.read_csv('data/houseprice/test.csv')
house_test = house_test2[["Id","BedroomAbvGr"]]
house_test

a = beta_1
b = beta_0
x = house_test["BedroomAbvGr"]
house_test["SalePrice"] = (a * x + b) * 1000

# submission 파일 만듬
submission2 = house_test[["Id","SalePrice"]]
submission2

submission2.to_csv("data/houseprice/sample_submission12.csv", index=False)
--------------------------------------------------------------
# 파일 추출 방법2
sample = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/sample_submission.csv")

sample["SalePrice"] = (house_test["BedroomAbvGr"] * 46.97049 + 63.31967) * 1000

sample.to_csv('C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/sample_submission.csv', index=False)

------------------------------------------------------
# 직선 성능 평가
a=46
b=63
# y_hat
y_hat = (a * house_train["BedroomAbvGr"] + b) * 1000
# y는 어디있는가
y= house_train["SalePrice"]

np.abs(y-y_hat) # 절대거리
np.sum((y-y_hat)**2) # 작을 수록 예측 잘한것

----------------------------------------------------------
!pip install scikit-learn

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 예시 데이터 (x와 y 벡터)
x = np.array([1, 3, 2, 1, 5]).reshape(-1, 1)  # x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
y = np.array([1, 2, 3, 4, 5])  # y 벡터 (레이블 벡터는 1차원 배열입니다)

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기 및 절편 값 구해줌

# 회귀 직선의 기울기와 절편
slope = model.coef_[0]
intercept = model.intercept_
plt.rcParams.update({"font.family":"Malgun Gothic"})
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")

# 예측값 계산
y_pred = model.predict(x)

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='실제 데이터')
plt.plot(x, y_pred, color='red', label='회귀 직선')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()

------------------------------------------------------------
# 
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

my_df=house_train[["BedroomAbvGr", "SalePrice"]]
my_df["SalePrice"]=my_df["SalePrice"]/1000

x = np.array(my_df["BedroomAbvGr"]).reshape(-1, 1)
y = np.array(my_df["SalePrice"])

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y)

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

# 회귀 직선의 기울기와 절편
slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")

# 예측값 계산
y_pred = model.predict(x)

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='실제 데이터')
plt.plot(x, y_pred, color='red', label='회귀 직선')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()

-------------------------------------------------------------
