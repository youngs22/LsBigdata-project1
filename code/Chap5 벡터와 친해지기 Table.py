#랜덤하게 수 만들기
?np.random.randint

a = np.random.randint(1, 21, 10)
print(a)

# 수 고정되서 모두 같은 랜덤값 생김
np.random.seed(42) #파이썬에서 시드 42로 많이 고정함
a = np.random.randint(1, 21, 10)
print(a)

np.random.seed(2024)
a = np.random.randint(1, 21, 10)
print(a)

a[-2] # 맨끝에서 두번째 값
a[::2] # 스텝자리가 있는 것
a[0:5:2]

# 1에서 부터 1000사이 3의 배수의 합은?
sum(np.arange(3,1001,3))
x=np.arange(1,1001)
sum(x[2:1000:3])

sum(x[x%3==0])
y=x[x%3==0]
sum(y)

# 논리값의 True인 값만 다시 벡터 만듬
a[a>3]

np.random.seed(2024)
a=np.random.randint(1,10000,5)
# a[조건을 만족하는 논리형 벡터]
a[(a>2000) & (a<5000)]
a
a>2000
a<5000

# 
import pydataset
df=pydataset.data('mtcars')
df
np_df=np.array(df["mpg"])
np_df

# 15이상 25이하인 데이터 개수는?
sum((np_df>=15)&(np_df<=25)) # TURE 개수 세기
sum(np_df[(np_df>=15)&(np_df<=25)]) # 원소 다 더하기

# 평균 MPG보다 큰 자동차 댓수
sum(np_df<np.mean(np_df))

# mpg가 15보다 작거나 22이상 큰 자동차 댓수
sum((np_df<15)|(np_df>=22))

#
np.random.seed(2024)
a=np.random.randint(1,10000,5)
b=np.array(["A","B","C","F","W"])
a[(a>2000) & (a<5000)]
b[(a>2000) & (a<5000)]

# 15 이상 20이하인 자동차 모델
model_names=np.array(df.index)
model_names[(np_df>=15)&(np_df<=25)]

# mpg가 평균 이하인 자동차 모델
model_names[(np_df<np.mean(np_df))]

#
np.random.seed(2024)
a=np.random.randint(1,10000,5)
a[a>3000] = 3000
a

np.random.seed(2024)
a=np.random.randint(1,100,10)
a
a<50

# 처음으로 500보다 큰 숫자가 나오는 위치? 숫자는?
np.random.seed(2024)
a=np.random.randint(1,26346,1000)
a[a>5000][0]
a[np.where(a>5000)]
a[np.where(a>10000)][0]

# for i in range(0,1000,1):
#     np.random.seed(2024)
#     a=np.random.randint(1,26346,1000)
#     if a[i]>5000 :
#          print(a[i])
#          print("위치",a)
#          break

# 처음으로 24000보다 큰 숫자가 나오는 위치? 숫자는?
np.random.seed(2024)
a=np.random.randint(1,26346,1000)
x=np.where(a>24000)
x
type(x)
my_index=x[0][0]
a[my_index]

# 1000보다 큰 숫자들 중 50번째 위치한 숫자는?
np.random.seed(2024)
a=np.random.randint(1,26346,1000)
x=np.where(a>10000)
x_index=x[0][49]  #1000보다 큰 숫자들 중 50번째 숫자 위치 나옴
x_index
a[x[0][49]] # 그 위치로 a에서 숫자 찾기

# 500보다 작은 숫자들 중 가장 마지막을 나오는 숫자
np.random.seed(2024)
a=np.random.randint(1,26346,1000)
x=np.where(a<500)
x[0][-1]
a[x[0][-1]]

# 
a = np.array([20, np.nan, 13, 24, 309])
a + 3 # nan은 숫자 아니므로 더해지지 않음
np.mean(a) #nan있으면 평균 안나옴

np.nanmean(a) #non무시하고 평균구함
np.nan_to_num(a,nan=0)

#
a=None
b=np.nan

a+1
b+1

#np.isnan(a) : 벡터 a의 원소가 nan이면 TRUE
a_filtered = a[~np.isnan(a)] # nan이 아닌 수들로 a를 채움
a_filtered

# 리스트는 데이터 형식 섞여도 되지만, 벡터는 한가지 데이터형만 가능
list=["사과", 12, "수박", "참외"]
list

str_vec = np.array(["사과", "배", "수박", "참외"])
str_vec
str_vec[[0, 2]]

mix_vec = np.array(["사과", 12, "수박", "참외"], dtype=str)
mix_vec

combined_vec = np.concatenate((str_vec, mix_vec))
combined_vec

#
col_stacked = np.column_stack((np.arange(1, 5), np.arange(12, 16)))
col_stacked

row_stacked = np.vstack((np.arange(1, 5), np.arange(12, 16)))
row_stacked #np.row_stack이 아닌 vstack이 권장됨

#
uneven_stacked = np.column_stack((np.arange(1, 5), np.arange(12, 18)))
uneven_stacked

vec1 = np.arange(1, 5)
vec2 = np.arange(12, 18)
vec1 = np.resize(vec1, len(vec2)) #np.resize() 길이 강제로 맞춤
vec1
vec2
uneven_stacked = np.column_stack((vec1,vec2))
uneven_stacked

#연습문제1
a = np.array([1, 2, 3, 4, 5])
a+5

#연습문제2
a = np.array([12, 21, 35, 48, 5])
a[::2]

#연습문제3
a = np.array([1, 22, 93, 64, 54])
a.max()

#연습문제4
a = np.array([1, 2, 3, 2, 4, 5, 4, 6])
np.unique(a) # 중복값 제거한 새로운 백터 생성 np.unique()

#연습문제5
a = np.array([21, 31, 58])
b = np.array([24, 44, 67])
c= np.empty(6)

for i in range(0,6,2):
    c[i]=a[i]
    c[i+1]=b[i]
c
    
x=np.empty(6)
x
x[1::2]=a
x[0::2]=b
x
