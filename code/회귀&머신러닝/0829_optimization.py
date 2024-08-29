# (x-2)**2 +1 그래프 그리기
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-4, 8, 100)
y = (x-2)**2 +1

# plt.scatter(x,y,s=3)
plt.plot(x,y,color="black")

y_2 = 4*x-11
plt.plot(x,y_2,color="red")

plt.xlim(-4,8)
plt.ylim(0,15)

k=4
# f'(x)=2x-4 
# f(x)=(x-2)**2 +1
# k=4의 기울기
l_slope=2*k - 4 # a 기울기, f'(x)에 K값 넣은 y값
f_k=(k-2)**2 + 1 # f(x)에 K값 넣은 y값
l_intercept=f_k - l_slope * k # b = y - ax

# y=slope*x+intercept 그래프
line_y=l_slope*x + l_intercept

plt.plot(x, line_y, color="blue")

x = np.linspace(-4, 8, 100)
y = x**2
y_2 = 2*x


# 경사하강법
x=10
lstep=0.9
for i in range(1,101):
    x= x-lstep *(2*x)
    print(i, ":", x)

    
x=10
lstep=np.arange(100,0,-1)*0.01
for i in range(1,101):
    x= x-lstep *(2*x)
    print(x)
