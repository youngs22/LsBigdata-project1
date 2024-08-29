import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error

# 데이터 로드(1번)
berry_train = pd.read_csv('./data/blueberry/train.csv')
berry_test = pd.read_csv('./data/blueberry/test.csv')
sub_df = pd.read_csv('./data/blueberry/sample_submission.csv')

# # Nan 채우기
# 각 숫자 변수는 평균으로 채우기
# 각 범주형 변수는 최빈값으로 채우기
berry_train.isna().sum()
berry_test.isna().sum()

train_x = berry_train.drop("yield", axis=1)
train_y = berry_train["yield"]

test_x = berry_test.drop("yield", axis=1, errors='ignore')


# Validation(교차검증) 만들기(10번)
kf = KFold(n_splits=20, shuffle=True, random_state=2024)

# valid_result 한 칸 계산하는 함수
def rmse(model):
    # -- -> + 로 만들어준 것
    score = np.sqrt(-cross_val_score(model, 
                                     train_x, 
                                     train_y, 
                                     cv = kf,
                                     n_jobs=-1, 
                                     scoring = "neg_mean_squared_error").mean())
    return(score)

alpha_values = np.arange(0.007, 0.008, 0.00001)
# 막대기 만들어줌(valid_result)
mean_scores = np.zeros(len(alpha_values))

k=0
# for문으로 칸 다 채워서 막대기 채움
for alpha in alpha_values:
    # max_iter(정확하게 구하려고 정해줌)
    lasso = Lasso(alpha=alpha)
    # val_result
    # neg_mean_squared_error: 오차의 제곱
    mean_scores[k] = rmse(lasso)
    k += 1

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

# 결과 시각화
plt.plot(df['lambda'], df['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Lasso Regression Train vs Validation Error')
plt.show()

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)

# 선형 회귀 모델 생성(8번)
# model = Lasso(alpha=0.0079)
model = Lasso(alpha=0.00672)

# 모델 학습
model.fit(train_x, train_y) # train 적용

pred_y=model.predict(test_x) # test로 predict 하기

sub_df["yield"] = pred_y

# csv 파일로 내보내기
sub_df.to_csv("sample_submission35.csv", index=False)

# 블루베리 선형회귀---------------------------------------
import pandas as pd

# house_test = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/test.csv")
# sample = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/sample_submission.csv")
# house_train = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/train.csv")

# house_train = house_train.dropna(subset=["GarageArea","SalePrice"])
# house_test["GarageArea"].mean()
# house_test["GarageArea"] = house_test["GarageArea"].fillna(472.7)

# x = np.array(house_train["GarageArea"]).reshape(-1, 1)
# y = np.array(house_train["SalePrice"])/1000

from sklearn.linear_model import LinearRegression

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(train_x, train_y)

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

# 예측값 계산
train_x = berry_train.drop("yield", axis=1)
train_y = berry_train["yield"]

test_x = berry_test

pred_y = model.predict(test_x)

sub_df["yield"] = pred_y

sub_df.to_csv("sample_submission_linear.csv", index=False)

#------------------------
# KNN 활용
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import make_scorer, mean_squared_error

# 데이터 로드(1번)
berry_train = pd.read_csv('./data/blueberry/train.csv')
berry_test = pd.read_csv('./data/blueberry/test.csv')
sub_df = pd.read_csv('./data/blueberry/sample_submission.csv')

train_x = berry_train.drop("yield", axis=1)
train_y = berry_train["yield"]

test_x = berry_test.drop("yield", axis=1, errors='ignore')

# Validation(교차검증) 만들기
kf = KFold(n_splits=50, shuffle=True, random_state=2024)

# valid_result 한 칸 계산하는 함수
def rmse(model):
    # -- -> + 로 만들어준 것
    score = np.sqrt(-cross_val_score(model, 
                                     train_x, 
                                     train_y, 
                                     cv = kf,
                                     n_jobs=-1, 
                                     scoring = "neg_mean_squared_error").mean())
    return(score)

# KNN 모델에서 사용할 k 값을 테스트하기 위한 범위 설정
k_values = np.arange(5,10)  
# 막대기 만들어줌(valid_result)
mean_scores = np.zeros(len(k_values))

k_index = 0
# for문으로 칸 다 채워서 막대기 채움
for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    mean_scores[k_index] = rmse(knn)
    k_index += 1

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'k': k_values,
    'validation_error': mean_scores
})

# 결과 시각화
plt.plot(df['k'], df['validation_error'], label='Validation Error', color='blue')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('KNN Regression Train vs Validation Error')
plt.show()

# 최적의 k 값 찾기
optimal_k = df['k'][np.argmin(df['validation_error'])]
print("Optimal k:", optimal_k)

# 최적의 k를 사용한 KNN 모델 생성
model = KNeighborsRegressor(n_neighbors=optimal_k)

# 모델 학습
model.fit(train_x, train_y)

# 예측
pred_y = model.predict(test_x)

# 결과를 submission 파일로 저장
sub_df["yield"] = pred_y
sub_df.to_csv("sample_submission_knn.csv", index=False)

# --------------------------------------
# KNN 활용
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import make_scorer, mean_squared_error

# 데이터 로드(1번)
berry_train = pd.read_csv('./data/blueberry/train.csv')
berry_test = pd.read_csv('./data/blueberry/test.csv')
sub_df = pd.read_csv('./data/blueberry/sample_submission.csv')

train_x = berry_train.drop(["yield","id"], axis=1)
train_y = berry_train["yield"]

test_x = berry_test.drop(["yield","id"], axis=1, errors='ignore')

# Validation(교차검증) 만들기
kf = KFold(n_splits=60, shuffle=True, random_state=2024)

# valid_result 한 칸 계산하는 함수
def rmse(model):
    # -- -> + 로 만들어준 것
    score = np.sqrt(-cross_val_score(model, 
                                     train_x, 
                                     train_y, 
                                     cv = kf,
                                     n_jobs=-1, 
                                     scoring = "neg_mean_squared_error").mean())
    return(score)

# KNN 모델에서 사용할 k 값을 테스트하기 위한 범위 설정
k_values = np.arange(1,80)  
# 막대기 만들어줌(valid_result)
mean_scores = np.zeros(len(k_values))

k_index = 0
# for문으로 칸 다 채워서 막대기 채움
for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=int(k))
    mean_scores[k_index] = rmse(knn)
    k_index += 1

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'k': k_values,
    'validation_error': mean_scores
})

# 결과 시각화
plt.plot(df['k'], df['validation_error'], label='Validation Error', color='blue')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('KNN Regression Train vs Validation Error')
plt.show()

# 최적의 k 값 찾기
optimal_k = df['k'][np.argmin(df['validation_error'])]
print("Optimal k:", optimal_k)

# 최적의 k를 사용한 KNN 모델 생성
model = KNeighborsRegressor(n_neighbors=int(optimal_k))

# 모델 학습
model.fit(train_x, train_y)

# 예측
pred_y = model.predict(test_x)

# 결과를 submission 파일로 저장
sub_df["yield"] = pred_y
sub_df.to_csv("sample_submission_knn.csv", index=False)
