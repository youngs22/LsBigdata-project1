# 필기
# 2번 가로벡터 x 세로벡터
import numpy as np

a= np.arange(1,4)
b= np.array([3,6,9])

a.dot(b)

# 행렬 x 벡터
a= np.array([1,2,3,4]).reshape((2,2), order="F") # 세로로 채움
a

b= np.array([5,6]).reshape(2,1)
b

a.dot(b)
a@b  #곱하기가 @

# 행렬 x 행렬
a=np.array([1,2,3,4]).reshape((2,2), order="F")
b=np.array([5,6,7,8]).reshape((2,2), order="F")

a@b

# 행렬 x 행렬 Q1
a=np.array([1,2,1,0,2,3]).reshape((2,3))
b=np.array([1,-1,2,0,1,3]).reshape((3,2), order="F")
a@b

# 행렬 x 행렬 Q2
np.eye(3) #단위벡터
a=np.array([3,5,7,
            2,4,9,
            3,1,0]).reshape(3,3)
a
a @ np.eye(3)

# transpose
a.transpose()
b=a[:,0:2]  
b
b.transpose( )

# 부리길이 ~ 부리깊이, 날개길이 -> 여러개 한번에 계산
x=np.array([13,15,
            12,14,
            10,11,
            5,6]).reshape(4,2)
x
vec1=np.repeat(1,4).reshape(4,1)
vec1
matX=np.hstack((vec1,x)) # 옆으로 벡터 붙임
matX

beta_vec=np.array([2,3,1]).reshape(3,1)
beta_vec
matX @ beta_vec

# beta_vec에 따라서 잔차(y - matX @ beta_vec).transpose() @ (y - matX @ beta_vec) 바뀜
y=np.array([20,19,20,12]).reshape(4,1)
y
beta_vec=np.array([2,0,1]).reshape(3,1)
(y - matX @ beta_vec).transpose() @ (y - matX @ beta_vec)

# 역행렬
a= np.array([1,5,3,4]).reshape(2,2)
a_inv = (-1/11)* np.array([4,-5,-3,1])

# 3 by 3 역행렬
a=np.array([-4,-6,2,
            5,-1,3,
            -2,4,-3]).reshape(3,3)
a
a_inv=np.linalg.inv(a)
a_inv
np.round(a@a_inv,3)
np.linalg.det(a) # 선형독립이면 행렬식 0아님

# 선형종속
b=np.array([1,2,3,
            2,4,5,
            3,6,7]).reshape(3,3)
b
b_inv=np.linalg.inv(b)
b_inv=np.linalg.det(b) # 선형독립이면 행렬식 0임
b_inv

# 1. 벡터 형태로 베타 구하기
XtX_inv=np.linalg.inv((matX.transpose() @ matX))
Xty=matX.transpose() @ y
beta_hat=XtX_inv @ Xty
beta_hat

# 2. 모델 fit으로 베타 구하기
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(matX[:,1:],y)

model.coef_      # 기울기 a
model.intercept_ # 절편 b

# 3. minimize로 베타 구하기
from scipy.optimize import minimize

def line_perform(beta):
    beta=np.array(beta).reshape(3, 1)
    a=(y - matX @ beta)
    return (a.transpose() @ a)

line_perform([2, 0, 1])

# 초기 추정값
initial_guess = [0, 0, 0]

# 최소값 찾기
result = minimize(line_perform, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)


# minimize로 라쏘 베타 구하기
from scipy.optimize import minimize
import numpy as np

y = np.array([20,19,20,12]).reshape(4,1)
x = np.array([13,15,
            12,14,
            10,11,
            5,6]).reshape(4,2)
vec1=np.repeat(1,4).reshape(4,1)
matX=np.hstack((vec1,x))

def line_perform_lasso(beta):
    beta=np.array(beta).reshape(3, 1)
    a=(y - matX @ beta) # 잔차
    return (a.transpose() @ a) + 500 * np.abs(beta[1:]).sum()
# (a.transpose() @ a)는 잔차의 제곱합, 500이 람다, np.abs(beta[1:]).sum()는 베타의 제곱합

line_perform_lasso([8.14, 0.96, 0])

initial_guess = [0, 0, 0]

result = minimize(line_perform_lasso, initial_guess) #line_perform_lasso를 최소화하는 값 구함

print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

# [8.55, 5.96, -4.38], 람다 0, np.abs(beta[1:]).sum()

# [8.14, 0.96, 0], 람다 3, 3*np.abs(beta[1:]).sum()
# 예측식: y_hat = 8.14 + 0.96 * X1 + 0 * X2

# [17.74, 0, 0] 람다 500로 변경, 500*np.abs(beta[1:]).sum()
# 예측식: y_hat = 8.14 + 0 * X1 + 0 * X2

# 람다 값에 따라 변수 선택 됨
# X 변수가 추가되면, trainX에서는 성능향상이 좋아짐
# X 변수가 추가되면, validX에서는 성능 좋아졌다가 나빠짐(오버피팅)
# 어느 순간 X 변수 추가하는 것을 멈추어야 함.
# 람다 0부터 시작: 내가 가진 모든 변수를 넣겠다!
# 점점 람다를 증가: 변수가 하나씩 빠지는 효과
# validX에서 가장 성능이 좋은 람다를 선택
# 변수가 선택됨을 의미

#(X^T X)^-1
# X의 칼럼에 선형 종속인 애들 있다: 다중공선성이 존재한다.

# --------------------------------------------------
# minimize로 릿지 베타 구하기
from scipy.optimize import minimize

y = np.array([20,19,20,12]).reshape(4,1)
x = np.array([13,15,
            12,14,
            10,11,
            5,6]).reshape(4,2)
vec1=np.repeat(1,4).reshape(4,1)
matX=np.hstack((vec1,x))

def line_perform_ridge(beta):
    beta=np.array(beta).reshape(3, 1)
    a=(y - matX @ beta) 
    return (a.transpose() @ a) + 3 * np.abs(beta**2).sum()

line_perform_ridge([8.55, 5.96, -4.38])
line_perform_ridge([3.76, 1.36, 0])

initial_guess = [0, 0, 0]

result = minimize(line_perform_ridge, initial_guess)

print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)