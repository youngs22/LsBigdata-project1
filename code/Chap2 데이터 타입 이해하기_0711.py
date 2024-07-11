# 데이터 타입 
x = 15.34
print(x, "는 ", type(x), "형식입니다.", sep=' ') # sep: ''이면 공백없이 붙여줘라 ' '이면 공백 넣어 붙여줘라

# 문자형 데이터 예제
a="Hello, world!"
b='python programming'

print(a, type(a))
print(b, type(b))

ml_str = """This is
a multi-line
string"""
print(a, type(a))
print(b, type(b))
print(ml_str, type(ml_str))

# 리스트 타입
fruits = ['apple', 'banana', 'cherry']
type(fruits)
fruits

mixed_list = [1,"Hello", [1,2,3]]
mixed_list

# 튜플형 타입
a_tp=(10, 20, 30)
type(a_tp)

a_int=10, 20, 30  #숫자 여러개 넣으면 튜플형으로 생성됨
type(a_int)

b_tp=(42,)
type(b_tp)

b_int=(42)
type(b_int)

print("좌표:", a)
print("단원소 튜플:", b)

# 인덱싱과 슬라이싱
a_list=[10,20,30,40,50,60,70]
a_list[0]=50
a_list[2]
a_list
a_list[2:] #앞에서 세번째부터 끝까지/해당 인덱스 이상
a_list[:3] #첫번째부터 세번째까지/해당 인덱스 미만
a_list[1:3] #해당 인덱스 이상 &미만

a_tp=(10,20,30,40,50,60,70,80)
a_tp[0]
a_tp[2]
a_tp[0]=50 #튜플은 변경 불가능, 다같이 코딩할 때 수정못하게할 수 있음
a_tp[0:3]

# 사용자 정의 함수
# 리스트로 반환
def min_max(numbers):
  return [min(numbers), max(numbers)]

A= [1,2,3,4,5]
result = min_max(A)
type(result)
print("Minimum and maximum:",result)

# 튜플로 반환
def min_max(numbers):
  return (min(numbers), max(numbers))

A= [1,2,3,4,5]
result = min_max(A)
type(result)
print("Minimum and maximum:",result)

# 딕셔너리
person = {
  "name":"john",
  "age":30,
  "city":'New York'
  
}
issac={
  "name":"이삭",
  "나이": 39,
  "사는 곳": ("미국","한국")
}
print("person: ",person)
print("issac: ", issac)

issac.get("사는 곳")[0]
issac_age=issac.get("나이")
type(issac_age)
type(issac.get("나이"))


# 집합형 데이터타입
fruits = {'apple', 'banana', 'cherry', 'apple'} #중복은 알아서 제거해줌
type(fruits)
fruits #집합은 순서 마음대로 나옴

empty_set=set() # 비어있는 집합 생성 가능
print("Empty set: ", empty_set)

empty_set.add("apple")
empty_set.add("apple")
empty_set.add("banana")
empty_set.add("cherry")
print("Empty set: ", empty_set)

05empty_set.remove("apple")
empty_set.discard("apple")

#집합 간 연산
other_fruits = {'berry', 'cherry'}
union_fruits = fruits.union(other_fruits)
intersection_fruits = fruits.intersection(other_fruits)
print("Union of fruits:", union_fruits)
print("Intersection of fruits:", intersection_fruits)

# 논리형 데이터 예제
p = True
q = False
print(p, type(p))
print(q, type(q))
print(p + p) # True는 1로, False는 0으로 계산됩니다.

a=3
if (a == 2):
print("a는 2와 같습니다.")
else:
print("a는 2와 같지 않습니다.")

# 숫자형을 문자열형으로 변환
num = 123
str_num = str(num)
print("문자열:", str_num, type(str_num))

# 문자열형을 숫자형(실수)으로 변환
num_again = float(str_num)
print("숫자형:", num_again, type(num_again))

# 리스트와 튜플 변환
lst = [1, 2, 3]
print("리스트:", lst)
tup = tuple(lst)
print("튜플:", tup)

# 집합을 딕셔너리로 변환
set_example = {'a', 'b', 'c'}
# 딕셔너리로 변환 시, 일반적으로 집합 요소를 키 또는 값으로 사용
dict_from_set = {key: True for key in set_example}
print("Dictionary from set:", dict_from_set)
