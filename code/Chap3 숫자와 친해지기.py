import math
x=4
math.sqrt(4)

sqrt_val = math.sqrt(16)
print("16의 제곱근은: ", sqrt_val)

exp_val=math.exp(5)
print("e^5의 값은: ", exp_val)

log_

# 정규분포 확률밀도함수(PDF) 계산
def normal_pdf(x, mu, sigma):
sqrt_two_pi = math.sqrt(2 * math.pi)
factor = 1 / (sigma * sqrt_two_pi)
return factor * math.exp(-0.5 * ((x - mu) / sigma) ** 2)

def normal_pdf(x, mu, sigma):
  part1=(sigma * (math.sqrt(2*math.pi)))**-1
  part2=math.exp(-(x-mu)**2/(2*sigma**2))
  return part1 * part2

# 파라미터
mu = 0
sigma = 1
x = 1

# 확률밀도함수 값 계산
pdf_value = normal_pdf(x, mu, sigma)
print("정규분포 확률밀도함수 값은:", pdf_value)

#
x = 2
y = 9
z = math.pi / 2
result = (x ** 2 + math.sqrt(y) + math.sin(z)) * math.exp(x)
print("계산된 수식 값은:", result)


# snippets 사용하기
def fname(input):
    contents
    return

import pandas as pd

import seaborn as sns

import numpy as np

    
  
