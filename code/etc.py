a=[1,2,3]
a
a[1]=4
a

# soft copy -> a바꾸면 b도 바뀜/ b바뀌면 a도 바뀜
a=[1,2,3]
b=a #a에 들어있는 [1,2,3]의 주소를 b에 넣어라
b
a[1]=4
b
b[0]=5
a

id(a)
id(b) 

#deep copy -> a바꿔도 b안바뀜
a=[1,2,3]
b=a[:]
b
a[1]=4
b

id(a)
id(b) 

