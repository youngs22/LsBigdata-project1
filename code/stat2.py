import numpy as np

# x=1,2,3,...,32
x=np.arange(33)

# E(x)
np.arange(33).sum()/33
(np.arange(33)**2).sum()/33

# y=(x-E(x))^2
(np.arange(33) - 16 )**2
np.unique((x - 16)**2)*(2/33) # -로 음수 있어서 0제외 경우 2개씩임 -> 2/33

# var 분산 / np.unique() : 
sum(np.unique((np.arange(33) - 16)**2)*(2/33))

# E(x^2)
x=np.arange(33)
sum(x**2 *(1/33))

# E(x^2)-E(X)^2 = 분산
sum(x**2 *(1/33)) - 16**2

# 1
( 2/6+((2**2)*(2/6))+((3**2)*(1/6)) ) - ( 2/6 + 2*(2/6) +3*(1/6) )**2

a=np.arange(4)
p=np.array([1/6,2/6,2/6,1/6])

## E(x) E(E^2)
Ex = sum(a * p)
Ex2 = sum(a**2 * p)
Ex2-Ex**2

#2
x_1_50_1 = np.concatenate((np.arange(1,51), np.arange(49,0,-1)))
pro_x = x_1_50_1/2500

Ex = sum(x_1_50_1 * pro_x)
Ex2 = sum(x_1_50_1**2 * pro_x)
Ex2-Ex**2

#3
x=np.range(4)*2
p=np.array([1/6,2/6,2/6,1/6])

Ex = sum(x * p)
Ex2 = sum(x**2 * p)
Ex2-Ex**2

# 
import numpy as np
np.sqrt(9.52**2 / 16)
np.sqrt(9.52**2 / 10)
