import pandas as pd
from sklearn.linear_model import LinearRegression
house_test = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/test.csv")
sample = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/sample_submission.csv")
house_train = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/train.csv")

house_train = house_train.query("GrLivArea<=4500")

## 회귀분석 적합하기
neighborhood_dumies = pd.get_dummies(
    house_train["Neighborhood"],
    drop_first = True
)
x = pd.concat([house_train[["GrLivArea","GarageArea"]], 
                neighborhood_dumies], axis=1)
y = house_train["SalePrice"]

model = LinearRegression()

model.fit(x, y)

model.coef_      # 기울기 a
model.intercept_ # 절편 b

neighborhood_dumies_test = pd.get_dummies(
    house_test["Neighborhood"],
    drop_first = True
)
x_test = pd.concat([house_test[["GrLivArea","GarageArea"]], 
                    neighborhood_dumies_test], axis=1)
                    
x_test["GrLivArea"].isna().sum()
x_test["GarageArea"].isna().sum()
neighborhood_dumies_test.isna().sum()
x_test = x_test.fillna(house_test["GarageArea"].mean())

pred_y = model.predict(x_test)

sample["SalePrice"] = pred_y

sample.to_csv('C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/sample_submission27.csv', index=False)

