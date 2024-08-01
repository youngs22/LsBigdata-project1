import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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
=============================================================
house

# 월별 이사 횟수 알아보기
df = house.dropna(subset=["MoSold","SalePrice"])\
                    .groupby("MoSold", as_index = False)\
                    .agg(count = ("SalePrice","count"))\
                    .sort_values("MoSold", ascending = True)
                    
sns.barplot(data=df, x="MoSold", y="count", hue="MoSold")
plt.rcParams.update({"font.family":"Malgun Gothic"})
plt.xlabel("월(month)")
plt.ylabel("이사횟수(count)")
plt.show()
plt.clf() 

==========================================
house_train3 = house[["BldgType", "OverallCond"]]

house_train3 = house_train3.dropna(subset=["BldgType","OverallCond"])\
                    .groupby(["OverallCond", "BldgType"], as_index = False)\
                    .agg(count = ("BldgType", "count"))\
                    .sort_values("count", ascending = False)
sns.barplot(data = house_train3, x = "OverallCond", y = "count", hue = "BldgType")
legend = plt.legend()
new_labels = ['단독주택', '타운하우스 끝 유닛', '듀플렉스', '2가구 개조 주택', '타운하우스 내부 유닛']  # 예시 이름
for text, new_label in zip(legend.get_texts(), new_labels):
    text.set_text(new_label)
plt.tight_layout()
plt.show()
plt.clf()


