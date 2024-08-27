import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

# Leave one out CV
# 20차 모델 만들기
# train으로 학습
np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

import pandas as pd
df = pd.DataFrame({
    "y" : y,
    "x" : x
})
df

train_df = df.loc[:19]
train_df

for i in range(2, 21):
    train_df[f"x{i}"] = train_df["x"] ** i

train_x = train_df[["x"] + [f"x{i}" for i in range(2, 21)]]
train_y = train_df["y"]

# -------------------------------------
valid_df = df.loc[20:]
valid_df

for i in range(2, 21):
    valid_df[f"x{i}"] = valid_df["x"] ** i

valid_x = valid_df[["x"] + [f"x{i}" for i in range(2, 21)]]
valid_x
valid_y = valid_df["y"]
valid_y

# 결과 받기 위한 벡터 만들기
val_result=np.repeat(0.0, 100)
tr_result=np.repeat(0.0, 100)

for i in np.arange(0, 100):
    model= Lasso(alpha=i*0.01)
    model.fit(train_x, train_y)

    # 모델 성능
    y_hat_train = model.predict(train_x)
    y_hat_val = model.predict(valid_x)

    perf_train=sum((train_df["y"] - y_hat_train)**2)
    perf_val=sum((valid_df["y"] - y_hat_val)**2)
    tr_result[i]=perf_train
    val_result[i]=perf_val

tr_result
val_result
# ------------------------------------

# seaborn을 사용하여 산점도 그리기
import seaborn as sns

df = pd.DataFrame({
    'lambda': np.arange(0, 1, 0.01), 
    'tr': tr_result,
    'val': val_result
})
sns.scatterplot(data=df, x='lambda', y='tr')
sns.scatterplot(data=df, x='lambda', y='val', color='red')
plt.xlim(0, 0.4)

val_result[0]
val_result[1]
np.min(val_result)

# alpha를 0.03로 선택!
np.argmin(val_result)
np.arange(0, 1, 0.01)[np.argmin(val_result)]

# 라쏘 활용
from sklearn.linear_model import Lasso

model= Lasso(alpha=0.03)
model.fit(train_x, train_y)
model.coef_
model.intercept_
# model.predict(test_x)

k=np.linspace(-4, 4, 800)

k_df = pd.DataFrame({
    "x" : k
})

for i in range(2, 21):
    k_df[f"x{i}"] = k_df["x"] ** i
    
k_df

reg_line = model.predict(k_df)

plt.plot(k_df["x"], reg_line, color="red")
plt.scatter(valid_df["x"], valid_df["y"], color="blue")

# ------------------------------------------------
# 3Ford
np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

import pandas as pd
df = pd.DataFrame({
    "y" : y,
    "x" : x
})
df

for i in range(2, 21):
    df[f"x{i}"] = df["x"] ** i

# myindex[0:10]
# myindex[10:20]
# myindex[20:30]

# 이런식으로 만들고 싶음
# train_set1=np.array([myindex[0:10], myindex[10:20]])
# valid_set1=myindex[20:30]

def make_tr_val(fold_num, df):
    np.random.seed(2024)
    myindex=np.random.choice(30, 30, replace=False)

    # valid index
    val_index=myindex[(10*fold_num):(10*fold_num+10)]

    # valid set, train set
    valid_set=df.loc[val_index]
    train_set=df.drop(val_index)

    train_X=train_set.iloc[:,1:]
    train_y=train_set.iloc[:,0]

    valid_X=valid_set.iloc[:,1:]
    valid_y=valid_set.iloc[:,0]

    return (train_X, train_y, valid_X, valid_y)


from sklearn.linear_model import Lasso

val_result_total=np.repeat(0.0, 3000).reshape(3, -1)
tr_result_total=np.repeat(0.0, 3000).reshape(3, -1)

for j in np.arange(0, 3):
    train_X, train_y, valid_X, valid_y = make_tr_val(fold_num=j, df=df)

    # 결과 받기 위한 벡터 만들기
    val_result=np.repeat(0.0, 1000)
    tr_result=np.repeat(0.0, 1000)

    for i in np.arange(0, 1000):
        model= Lasso(alpha=i*0.01)
        model.fit(train_X, train_y)

        # 모델 성능
        y_hat_train = model.predict(train_X)
        y_hat_val = model.predict(valid_X)

        perf_train=sum((train_y - y_hat_train)**2)
        perf_val=sum((valid_y - y_hat_val)**2)
        tr_result[i]=perf_train
        val_result[i]=perf_val

    tr_result_total[j,:]=tr_result
    val_result_total[j,:]=val_result


import seaborn as sns

df = pd.DataFrame({
    'lambda': np.arange(0, 10, 0.01), 
    'tr': tr_result_total.mean(axis=0),
    'val': val_result_total.mean(axis=0)
})

df['tr']

# seaborn을 사용하여 산점도 그리기
sns.scatterplot(data=df, x='lambda', y='tr')
sns.scatterplot(data=df, x='lambda', y='val', color='red')
plt.xlim(0, 10)

val_result[0]
val_result[1]
np.min(val_result_total.mean(axis=0))

# alpha를 2.67로 선택!
np.argmin(val_result_total.mean(axis=0))
np.arange(0, 10, 0.01)[np.argmin(val_result_total.mean(axis=0))]