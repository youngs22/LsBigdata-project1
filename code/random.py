import numpy as np
import pandas as pd

old_seat = np.arange(1,29)

np.random.seed(20240729)

# 1~28까지 숫자 중에서 중복 없이 28개 숫자를 뽑는 방법

new_seat = np.random.choice(old_seat, 28, replace=False)

result = pd.DataFrame({
        "old_seat" : old_seat,
        "new_seat" : new_seat
    })
    
result.to_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/result.csv", index=False)

# y=2x 그래프 그리기
import matplotlib.pyplot as plt
x=np.arange(50)
y=2 * x
plt.scatter(x,y,s=5, color="blue")
plt.plot(x,y,color="black")
plt.show()

# y=x^2를 점3개를 이용해서 그리기
plt.clf()
x=np.linspace(-8,8,3)
y=x**2
plt.scatter(x,y,s=5, color="blue")
plt.plot(x,y,color="black")
plt.show()

plt.clf()
x=np.linspace(-8,8,100)
y=x**2
# x, y측 범위 설정
plt.xlim(-10,10)
plt.ylim(0,40)
#비율맞추기
plt.gca().set_aspect("equal", adjustable="box")
# plt.scatter(x,y,s=5, color="blue")
plt.plot(x,y,color="black")
plt.show()

?plt.axis
