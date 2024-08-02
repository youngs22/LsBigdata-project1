# x^2+3 함수 만들기
def my_f(x):
    return x**2+3
my_f(3)

# z=x^2+y^2+3
# minimize는 무조건 x만 받기 때문에, 
# 변수가 2개라면 x를 리스트로 받아 x는 x의0번째, y는 x의1번째로 생각
def my_f2(x):
    return x[0]**2 + x[1]**2 + 3
my_f2([1,3])

# k=(x-1)^2 + (y-2)^2 + (z-4)^2
def my_f3(x):
    return (x[0]-1)**2 + (x[1]-2)**2 + (x[2]-4)**2 +7


# 최소값을 찾을 다변수 함수 정의
import numpy as np
from scipy.optimize import minimize

# 초기 추정값
initial_guess = [0,0,0] 

# 최소값 찾기
result = minimize(my_f3, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

-------------------------------------------------------------
# house 데이터 회귀분석모델
#1
import pandas as pd
from sklearn.linear_model import LinearRegression
house_test = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/test.csv")
sample = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/sample_submission.csv")
house_train = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/train.csv")

house_train = house_train.dropna(subset=["GarageArea","SalePrice"])
house_test["GarageArea"].mean()
house_test["GarageArea"] = house_test["GarageArea"].fillna(472.7)

x = np.array(house_train["GarageArea"]).reshape(-1, 1)
y = np.array(house_train["SalePrice"])/1000

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

test_x = np.array(house_test["GarageArea"]).reshape(-1, 1)

pred_y = model.predict(test_x)

sample["SalePrice"] = pred_y * 1000

sample.to_csv('C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/sample_submission15.csv', index=False)

----------------------------------------------------------------
#2
import pandas as pd
from sklearn.linear_model import LinearRegression
house_test = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/test.csv")
sample = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/sample_submission.csv")
house_train = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/train.csv")

house_train = house_train.dropna(subset=["LotArea","GarageArea","SalePrice"])
house_test["GarageArea"].mean()
house_test["GarageArea"] =   house_test["GarageArea"].fillna(472.7)

x = np.array(house_train[["GarageArea","LotArea"]]).reshape(-1, 2)
y = np.array(house_train["SalePrice"])/1000

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

test_x = np.array(house_test[["GarageArea","LotArea"]]).reshape(-1, 2)

pred_y = model.predict(test_x)

sample["SalePrice"] = pred_y * 1000

sample.to_csv('C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/sample_submission17.csv', index=False)

-----------------------------------------------------------------
#3 -> GrLivArea 제일 잘나옴
# 이상치 제거 전
import pandas as pd
from sklearn.linear_model import LinearRegression
house_test = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/test.csv")
sample = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/sample_submission.csv")
house_train = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/train.csv")

x = np.array(house_train["GrLivArea"]).reshape(-1, 1)
y = np.array(house_train["SalePrice"])/1000

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

pred_y = model.predict(x)

# 그래프그리기
plt.scatter(x, y, color='blue', label='실제 데이터')
plt.plot(x, pred_y, color='red', label='회귀 직선')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()


# 예측값 계산

test_x = np.array(house_test["GrLivArea"]).reshape(-1, 1)

pred_y = model.predict(test_x)

sample["SalePrice"] = pred_y * 1000

sample.to_csv('C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/sample_submission19.csv', index=False)
-------------------

# 이상치 제거 후
import pandas as pd
from sklearn.linear_model import LinearRegression
house_test = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/test.csv")
sample = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/sample_submission.csv")
house_train = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/train.csv")

# 이상치 탐색
house_train = house_train.query("GrLivArea<=4500")

x = np.array(house_train["GrLivArea"]).reshape(-1, 1)
y = house_train["SalePrice"]/1000

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

pred_y = model.predict(x)

# 그래프그리기
plt.scatter(x, y, color='blue', label='실제 데이터')
plt.plot(x, pred_y, color='red', label='회귀 직선')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()

# 예측값 계산

test_x = np.array(house_test["GrLivArea"]).reshape(-1, 1)

pred_y = model.predict(test_x)

sample["SalePrice"] = pred_y * 1000

sample.to_csv('C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/sample_submission19.csv', index=False)

------------------------------------------------------------
# x,y에 대괄호 하나면 판다스 시리즈로, 대괄호 두개면 판다스 프레임으로
# 판다스 시리즈는 차원이 없음, 그냥 length라고 뜸
# 판다스 프레임은 2차원으로 뜸
# 리스트로 되면 np.array와 .reshpae(-1,1) 안해줘도 됨
import pandas as pd
from sklearn.linear_model import LinearRegression
house_test = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/test.csv")
sample = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/sample_submission.csv")
house_train = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/train.csv")

house_train = house_train.query("GrLivArea<=4500")

x = house_train[["GrLivArea","GarageArea"]] # 대괄호 두개로 판다스 프레임(2차원)
y = house_train["SalePrice"]

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y)

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

# 회귀 직선의 기울기와 절편
slope = model.coef_
intercept = model.intercept_
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")

pred_y = model.predict(x)

# 그래프그리기
plt.scatter(x, y, color='blue', label='실제 데이터')
plt.plot(x, pred_y, color='red', label='회귀 직선')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()

# 회귀식 함수 만들기
def my_houseprice(x,y):
    return x*model.coef_[0] + y*model.coef_[1] + model.intercept_

my_houseprice(300,55)

# 회귀식에 넣은 값 도출
my_f4(house_train["GrLivArea"],house_train["GarageArea"])
-------------------------------------------------------------------

# test 값 예측하기
import pandas as pd
from sklearn.linear_model import LinearRegression
house_test = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/test.csv")
sample = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/sample_submission.csv")
house_train = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/train.csv")

house_train = house_train.query("GrLivArea<=4500")

x = house_train[["GrLivArea","GarageArea"]] # 대괄호 두개로 판다스 프레임(2차원)
y = house_train["SalePrice"]

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y)

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b


# 결측치 확인
test_x = house_test[["GrLivArea","GarageArea"]]

test_x["GrLivArea"].isna().sum()
test_x["GarageArea"].isna().sum()
test_x = test_x.fillna(house_test["GarageArea"].mean())

# test 값 예측
pred_y = model.predict(test_x)

sample["SalePrice"] = pred_y

sample.to_csv('C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/sample_submission20.csv', index=False)
------------------------------------------------------

# 변수 2개인거 3D 그래프 그리기
import pandas as pd
from sklearn.linear_model import LinearRegression
house_test = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/test.csv")
sample = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/sample_submission.csv")
house_train = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/train.csv")

house_train = house_train.query("GrLivArea<=4500")

x = house_train[["GrLivArea","GarageArea"]] # 대괄호 두개로 판다스 프레임(2차원)
y = house_train["SalePrice"]

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y)

# 기울기와 절편 출력
slope_grlivarea = model.coef_[0]
slope_garagearea = model.coef_[1]
intercept = model.intercept_

# 3D 그래프 그리기
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 데이터 포인트
ax.scatter(x['GrLivArea'], x['GarageArea'], y, color='blue', label='Data points')

# 회귀 평면
GrLivArea_vals = np.linspace(x['GrLivArea'].min(), x['GrLivArea'].max(), 100)
GarageArea_vals = np.linspace(x['GarageArea'].min(), x['GarageArea'].max(), 100)
GrLivArea_vals, GarageArea_vals = np.meshgrid(GrLivArea_vals, GarageArea_vals)
SalePrice_vals = intercept + slope_grlivarea * GrLivArea_vals + slope_garagearea * GarageArea_vals

ax.plot_surface(GrLivArea_vals, GarageArea_vals, SalePrice_vals, color='red', alpha=0.5)

# 축 라벨
ax.set_xlabel('GrLivArea')
ax.set_ylabel('GarageArea')
ax.set_zlabel('SalePrice')

plt.tight_layout()
plt.legend()
plt.show()
