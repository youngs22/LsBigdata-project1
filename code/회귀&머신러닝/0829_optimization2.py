import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# x, y의 값을 정의합니다 (-1에서 7까지)
x = np.linspace(-1, 7, 400)
y = np.linspace(-1, 7, 400)
x, y = np.meshgrid(x, y)

# 함수 f(x, y)를 계산합니다.
z = (x - 3)**2 + (y - 4)**2 + 3

# 그래프를 그리기 위한 설정
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 표면 그래프를 그립니다.
ax.plot_surface(x, y, z, cmap='viridis')

# 레이블 및 타이틀 설정
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(x, y)')
ax.set_title('Graph of f(x, y) = (x-3)^2 + (y-4)^2 + 3')

# 그래프 표시
plt.show()

# ==========================
# 등고선 그래프
import numpy as np
import matplotlib.pyplot as plt

# x, y의 값을 정의합니다 (-1에서 7까지)
x = np.linspace(-10, 10, 400)
y = np.linspace(-10, 10, 400)
x, y = np.meshgrid(x, y)

# 함수 f(x, y)를 계산합니다.
z = (x - 3)**2 + (y - 4)**2 + 3
z = (1-(x+y))**2 + (4-(x+2*y))**2 + (1.5-(x+3*y))**2 + (5-(x+4*y))**2

# 등고선 그래프를 그립니다.
plt.figure()
cp = plt.contour(x, y, z, levels=20)  # levels는 등고선의 개수를 조절합니다.
plt.colorbar(cp)  # 등고선 레벨 값에 대한 컬러바를 추가합니다.

# 특정 점 (9, 2)에 파란색 점을 표시
plt.scatter(9, 2, color='red', s=50)

x=9; y=2
lstep=0.1
for i in range(100):
    x, y = np.array([x, y]) - lstep * np.array([2*x-6, 2*y-8])
    plt.scatter(float(x), float(y), color='red', s=25)

print(x, y)

# 축 레이블 및 타이틀 설정
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Contour Plot of f(x, y) = (x-3)^2 + (y-4)^2 + 3')
plt.xlim(-10, 10)
plt.ylim(-10, 10)

# 그래프 표시
plt.show()

#-------------------------------------------
# 등고선 그래프
import numpy as np
import matplotlib.pyplot as plt

# x, y의 값을 정의합니다 (-1에서 7까지)
x = np.linspace(-10, 10, 400)
y = np.linspace(-10, 10, 400)
x, y = np.meshgrid(x, y)

# 함수 f(x, y)를 계산합니다.
z = (1-(x+y))**2 + (4-(x+2*y))**2 + (1.5-(x+3*y))**2 + (5-(x+4*y))**2

# 등고선 그래프를 그립니다.
plt.figure()
cp = plt.contour(x, y, z, levels=50)  # levels는 등고선의 개수를 조절합니다.
plt.colorbar(cp)  # 등고선 레벨 값에 대한 컬러바를 추가합니다.

x=10; y=10
lstep=0.01
for i in range(100):
    x, y = np.array([x, y]) - lstep * np.array([8*x+20*y-23, 60*y+20*x-67])
    plt.scatter(float(x), float(y), color='red', s=25)

print(x, y)

plt.xlim(-10, 10)
plt.ylim(-10, 10)

plt.show()
plt.clf()

#--------------------------------------
x = np.linspace(-10, 10, 400) # 400개
y = np.linspace(-10, 10, 400) # 400개
# 160000개 점
x, y = np.meshgrid(x, y)

# 함수 f(x, y)를 계산
z = (1-(x+y))**2 + (4-(x+2*y))**2 + (1.5-(x+3*y))**2 + (5-(x+4*y))**2

# 등고선 그래프를 그립니다.
plt.figure()
cp = plt.contour(x, y, z, levels=20)  # levels는 등고선의 개수를 조절
plt.colorbar(cp)  # 등고선 레벨 값에 대한 컬러바를 추가

x=10
y=10
lstep=0.1

for i in range(100):
     x, y = np.array([x, y]) - lstep * np.array([23-8*x, 23-20*y])
     plt.scatter(x, y, color="red", s=10)
x,y