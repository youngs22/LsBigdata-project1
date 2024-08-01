import numpy as np

# 두 개의 벡터를 합쳐 행렬 생성
matrix = np.column_stack((np.arange(1, 5),np.arange(12, 16)))
print("행렬:\n", matrix)

# 요소가 0인 행렬 만들기
np.zeros(5)
np.zeros([5,4])

np.arange(1,5).reshape((2,2))

# -1 통해서 크기를 자동으로 결정
np.arange(1,5).reshape((2,-1))

# 0~99까지 수중에서 랜덤하게 50개의 수 뽑아서 5 x 10 행렬을 만들기
np.random.seed(2024)
np.random.randint(0,100,50).reshape((5,10))
np.random.randint(0,100,50).reshape((5,-1))

# 가로 방향으로 채우기 (기본값)
y = np.arange(1, 5).reshape((2, 2), order='C')
print("가로 방향으로 채운 행렬 y:\n", y)

# 세로 방향으로 채우기
mat_a = np.arange(1, 21).reshape((4, 5), order='F')
print("세로 방향으로 채운 행렬 y:\n", mat_a)

# 인덱싱
mat_a[0,0]
mat_a[1,1]
mat_a[2,3]
mat_a[0:2,3] # 행 0,1에서 3과 만나는 지점
mat_a[1:3,1:4] 
mat_a[3,] # 열자리 비어있으면 전체 선택 
mat_a[3,:] # 행자리/열자리 : 있으면 전체 선택 
mat_a[3,::2] 

# 짝수 행만 선택하려면?
mat_b = np.arange(1, 101).reshape((20, -1))
mat_b
mat_b[1::2,]
mat_b[[0,4,6,19],]

# True에 해당하는 행 출력
x = np.arange(1, 11).reshape((5, 2)) * 2
x[[True, True, False, False, True], 0] 

# 벡터로 반환 -> 벡터는 1차원
mat_b[:,1]
mat_b[:,1].reshape((-1,1)) #reshape으로 행렬 형태 만듬

# 행렬로 반환 -> 행렬(matrix)은 2차원
mat_b[:,1:2]
mat_b[:,(1,)]
mat_b[:,(1,2)]

# 필터링
mat_b
mat_b[:,1]
mat_b[mat_b[:,1] % 7 == 0,:] #1번째 열에서 7의 배수가 있는 행전부 출력



# 사진 행렬
import numpy as np
import matplotlib.pyplot as plt

#난수 생성하여 3x3 크기의 행렬 생성
# np.random.rand : 0과1사이의 난수
np.random.seed(2024)
img1 = np.random.rand(3, 3)
print("이미지 행렬 img1:\n", img1)

plt.clf()
plt.imshow(img1, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()

a = np.random.randint(0,10,20).reshape(4,-1)
a / 9
plt.clf()
plt.imshow(a, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()

# 행렬 뒤집기
# 5행 2열의 행렬 생성
x = np.arange(1, 11).reshape((5, 2)) * 2
print("원래 행렬 x:\n", x)
# 행렬을 전치
transposed_x = x.transpose()
print("전치된 행렬 x:\n", transposed_x)

# 사진은 배열
# urllib url라이브러리의 request패키지 불러옴
import imageio.v2 as imageio
import urllib.request
img_url = "https://bit.ly/3ErnM2Q"
urllib.request.urlretrieve(img_url,"C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/img/jelly.png")

jelly = imageio.imread("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/img/jelly.png")
print("이미지 클래스:", type(jelly))
print("이미지 차원:", jelly.shape)
print("이미지 첫 4x4 픽셀, 첫 번째 채널:\n", jelly[:4, :4, 0])

# 앞 3장은 RGB이고 마지막장은 투명도 결정
jelly[:,:,0]
jelly[:,:,1]
jelly[:,:,2]
jelly[:,:,3]
jelly[:,:,0] # 0이 가장 투명

plt.imshow(jelly)
plt.axis('off')
plt.show()

# 옆으로 뉘우기
jelly.shape
jelly[:,:,0].shape
jelly[:,:,0].transpose().shape

plt.clf()

plt.imshow(jelly[:,:,0])
plt.imshow(jelly[:,:,1])
plt.imshow(jelly[:,:,2])
plt.imshow(jelly[:,:,3])

plt.axis('off')
plt.show()

# 사진은 배열2 - 다른 사진
# urllib url라이브러리의 request패키지 불러옴
import imageio.v2 as imageio
import urllib.request
img_url = "https://search.pstatic.net/common/?src=http%3A%2F%2Fblogfiles.naver.net%2FMjAyMjAxMTdfMTgy%2FMDAxNjQyMzgyMzUwNTc3.CVK5PFllic5ZT83I1nZhm1Ancs88FuK8v2yDkqRqdEYg.Ki5k6X4_EgonMstTnu-5gKq9xTT4n6OwJYMg42i0GMQg.JPEG.arji547%2F%25B5%25B6%25C0%25CF_%25B0%25A1%25C0%25BB%25C7%25B3%25B0%25E6_%252814%2529.jpg&type=sc960_832"
urllib.request.urlretrieve(img_url,"C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/img/house.png")

house = imageio.imread("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/img/house.png")
print("이미지 클래스:", type(house))
print("이미지 차원:", house.shape)
print("이미지 첫 4x4 픽셀, 첫 번째 채널:\n", house[:4, :4, 0])

# 앞 3장은 RGB이고 마지막장은 투명도 결정
house[:,:,0]
house[:,:,1]
house[:,:,2]
house[:,:,3]
house[:,:,0] # 0이 가장 투명

plt.imshow(house)
plt.axis('off')
plt.show()

# 옆으로 뉘우기
jelly[:,:,0].shape
jelly[:,:,0].transpose().shape

plt.clf()

plt.imshow(house[:,:,0])
plt.imshow(house[:,:,1])
plt.imshow(house[:,:,2])
plt.imshow(house[:,:,3])

plt.axis('off')
plt.show()

# 3차원 배열
# 두 개의 2x3 행렬 생성
mat1 = np.arange(1, 7).reshape(2, 3)
mat2 = np.arange(7, 13).reshape(2, 3)
mat1
mat1.shape
mat2
mat2.shape

# 3차원 배열로 합치기
my_array = np.array([mat1, mat2])
my_array
my_array.shape

# 3차원 배열 인덱싱
first_slice = my_array[0, :, :]
first_slice

filtered_array = my_array[:, :, :-1]
filtered_array

my_array[:,0,:]
my_array[0,1,1:3]
my_array[0,1,[1,2]]

#
mat_x = np.arange(1,101).reshape((5,5,4))
mat_x

mat_y = np.arange(1,101).reshape((2,2,25))
mat_y

# 넘파이 배열 메서드
a = np.array([[1,2,3],[4,5,6]])
a
a.sum()
a.sum(axis=0) #axis=0은 열기준
a.sum(axis=1) #axis=1은 행기준
a.mean(axis=0)
a.mean(axis=1)

b=np.random.randint(0,100,50).reshape(5,-1)
b
b.max()

# 행별 가장 큰수
b.max(axis=1)

# 열별 가장 큰수
b.max(axis=0)

a=np.array([1,3,2,5]).reshape((2,2))
a
a.cumsum() # 누적합 한 행으로 나옴
a.cumsum(axis=1) # 행별 누적합

b.cumsum(axis=1)

b.cumprod() # 누적곱

b.reshape((2,5,5)).shape
b.flatten() # 1차원으로 만들어줌, 쭉 펴줌

d=np.array([1,2,3,4,5])
d.clip(2,4)

c=d.tolist()
type(c)


