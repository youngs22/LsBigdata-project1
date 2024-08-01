import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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
# 직선의 방정식
# y+ ax + b
# 예: y = 2x+3 의 그래프를 그려보세요!
import numpy as np
import pandas as pd

# 예제 데이터
x = my_df["BedroomAbvGr"]
y = my_df["SalePrice"]

# 평균 계산
x_mean = np.mean(x)
y_mean = np.mean(y)

# Sxx 계산
Sxx = np.sum((x - x_mean) * (x - x_mean))

# Sxy 계산
Sxy = np.sum((x - x_mean) * (y - y_mean))

# 결과 출력
print(f"Sxx = {Sxx}")
print(f"Sxy = {Sxy}")

# 회귀 계수 계산
beta_1 = Sxy / Sxx
beta_0 = y_mean - beta_1 * x_mean

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
#각 년도별 평균 집값 구함

#test 파일에서 id랑년도만 빼줌
house_test = pd.read_csv('data/houseprice/test.csv')
house_test = house_test2[["Id","BedroomAbvGr"]]
house_test

a = beta_1
b = beta_0
x = house_test["BedroomAbvGr"]
house_test["SalePrice"] = (a * x + b) * 1000


# # 결측치 처리
# house_test2["SalePrice"].isna().sum()
# 
# price_mean = house["SalePrice"].mean()
# price_mean
# 
# house_test2 = house_test2.fillna(price_mean)

# submission 파일 만듬
submission2 = house_test[["Id","SalePrice"]]
submission2

submission2.to_csv("data/houseprice/sample_submission12.csv", index=False)
--------------------------------------------------------------
sample = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/sample_submission.csv")

sample["SalePrice"] = (house_test["BedroomAbvGr"] * 46.97049 + 63.31967) * 1000

sample.to_csv('C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/sample_submission.csv', index=False)
