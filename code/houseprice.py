import pandas as pd
import numpy as np


house = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/train.csv")
house.shape
house["SalePrice"].mean()

sample = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/sample_submission.csv")
sample["SalePrice"] = house["SalePrice"].mean()
sample

sample.to_csv('C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/houseprice/sample_submission.csv', index=False)
