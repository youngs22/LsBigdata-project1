mport pandas as pd
import numpy as np

house = pd.read_csv('data/houseprice/train.csv')

sub.to_csv("data/houseprice/sample_submission.csv", index=False)
#인덱스 없애고자 할 때, index=False로 해줌


#같은 해에 지어진 그룹을 한 그룹으로 보고 ->평균을 냄
#test.set에 있는 집값을 예측해보자.

#각 년도별 평균 집값 구함
house_mean=house.groupby("YearBuilt", as_index=False)\
         .agg( 
             house_mean=("SalePrice", "mean")
         )

#test 파일에서 id랑년도만 빼줌
house_test = pd.read_csv('data/houseprice/test.csv')
house_test = house_test[["Id","YearBuilt"]]
house_test

# test 파일에 집값 평균 구한거 연도에 맞게 추가
house_test = pd.merge(house_test, house_mean, how = "left", on = "YearBuilt")
house_test

# 열이름 바꿈
house_test = house_test.rename(
    columns = {"house_mean" : "SalePrice"}
)
house_test

# 결측치 처리
house_test["SalePrice"].isna().sum()

price_mean = house["SalePrice"].mean()
price_mean

house_test = house_test.fillna(price_mean)

# submission 파일 만듬
submission = house_test[["Id","SalePrice"]]
submission

submission.to_csv("data/houseprice/sample_submission2.csv", index=False)

======================================================================================
import pandas as pd
import numpy as np

house = pd.read_csv('data/houseprice/train.csv')

sub.to_csv("data/houseprice/sample_submission.csv", index=False)
#인덱스 없애고자 할 때, index=False로 해줌


#같은 해에 지어진 그룹을 한 그룹으로 보고 ->평균을 냄
#test.set에 있는 집값을 예측해보자.

#각 년도별 평균 집값 구함
house_mean2=house.groupby(["YearBuilt","GarageCars","KitchenQual"], as_index=False)\
         .agg( 
             house_mean=("SalePrice", "mean")
         )
house_mean2

#test 파일에서 id랑년도만 빼줌
house_test2 = pd.read_csv('data/houseprice/test.csv')
house_test2 = house_test2[["Id","YearBuilt","GarageCars","KitchenQual"]]
house_test2

# test 파일에 집값 평균 구한거 연도에 맞게 추가
house_test2 = pd.merge(house_test2, house_mean2, how = "left", on = ["YearBuilt","GarageCars","KitchenQual"])
house_test2

# 열이름 바꿈
house_test2 = house_test2.rename(
    columns = {"house_mean" : "SalePrice"}
)
house_test2

# 결측치 처리
house_test2["SalePrice"].isna().sum()

price_mean = house["SalePrice"].mean()
price_mean

house_test2 = house_test2.fillna(price_mean)

# submission 파일 만듬
submission2 = house_test2[["Id","SalePrice"]]
submission2

submission2.to_csv("data/houseprice/sample_submission3.csv", index=False)

=================================================================================
#각 년도별 평균 집값 구함
house_mean2=house.groupby(["YearBuilt","Exterior1st","Foundation"], as_index=False)\
         .agg( 
             house_mean=("SalePrice", "mean")
         )
house_mean2

#test 파일에서 id랑년도만 빼줌
house_test2 = pd.read_csv('data/houseprice/test.csv')
house_test2 = house_test2[["Id","YearBuilt","Exterior1st","Foundation"]]
house_test2

# test 파일에 집값 평균 구한거 연도에 맞게 추가
house_test2 = pd.merge(house_test2, house_mean2, how = "left", on = ["YearBuilt","Exterior1st","Foundation"])
house_test2

# 열이름 바꿈
house_test2 = house_test2.rename(
    columns = {"house_mean" : "SalePrice"}
)
house_test2

# 결측치 처리
house_test2["SalePrice"].isna().sum()

price_mean = house["SalePrice"].mean()
price_mean

house_test2 = house_test2.fillna(price_mean)

# submission 파일 만듬
submission2 = house_test2[["Id","SalePrice"]]
submission2

submission2.to_csv("data/houseprice/sample_submission5.csv", index=False)
