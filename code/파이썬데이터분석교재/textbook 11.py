#11-1
import json
geo = json.load(open("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/SIG.geojson", encoding="UTF-8"))
geo

geo["features"][0]["properties"]
geo["features"][0]["geometry"]

import pandas as pd
df_pop = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/Population_SIG.csv")
df_pop
df_pop.info()

df_pop["code"]=df_pop["code"].astype(str)
df_pop["code"]

?pip install folium
import folium

folium.Map(location=[35.95,127.7],
            zoom_start= 8)

map_sig = folium.Map(location=[35.95,127.7],
                    zoom_start= 8,
                    titles="cartobposition")
map_sig

folium.Choropleth(
    geo_data = geo,
    data=df_pop,
    columns = ("code","pop"),
    key_on = "feature.properties.SIG_CD")\
    .add_to(map_sig)
------------------------------------------------
# 11-2
import json
import pandas as pd
geo_seoul = json.load(open("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/SIG_Seoul.geojson", encoding="UTF-8"))

type(geo_seoul)
len(geo_seoul)
geo_seoul.keys()
geo_seoul["features"][0]
len(geo_seoul["features"])
len(geo_seoul["features"][0])
geo_seoul["features"][0].keys()
geo_seoul["features"][0]["geometry"]
geo_seoul["features"][0]["properties"]

coordinate_list=geo_seoul["features"][2]["geometry"]["coordinates"]
len(coordinate_list)
len(coordinate_list[0])
len(coordinate_list[0][0])
coordinate_list
coordinate_list[0]
coordinate_list[0][0]

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

coordinate_array = np.array(coordinate_list[0][0])
x=coordinate_array[:,0]
y=coordinate_array[:,1]
plt.scatter(x,y,s=3)
plt.show()
plt.clf()
------------------------------------------------------------
# (11-2)서울시 구 추출 모델
def draw_seoul(num):
    coordinate_list=geo_seoul["features"][num]["geometry"]["coordinates"]
    gu_name=geo_seoul["features"][num]["properties"]["SIG_KOR_NM"]
    coordinate_array = np.array(coordinate_list[0][0])
    x=coordinate_array[:,0]
    y=coordinate_array[:,1]
    plt.rcParams['font.family'] ='Malgun Gothic'
    plt.plot(x,y)
    plt.title(gu_name+"의 지도")
    plt.show()
    plt.clf()

draw_seoul(12)

----------------------------------------------------------
# 서울시 구별로 구역 색 다르게 지도 그리기
# gpt
data = []

# geo_seoul의 모든 구에 대해 반복
for i in range(len(geo_seoul['features'])):
    # 좌표 및 구 이름 추출
    coordinate_list_all = geo_seoul['features'][i]['geometry']['coordinates']
    coordinate_array = np.array(coordinate_list_all[0][0])
    gu_name = geo_seoul['features'][i]['properties']['SIG_KOR_NM']
    
    # 좌표와 구 이름을 데이터 리스트에 추가
    for coord in coordinate_array:
        data.append({'gu_name': gu_name, 'x': coord[0], 'y': coord[1]})
df = pd.DataFrame(data)

from shapely.geometry import Point
import geopandas as gpd

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs="EPSG:4326")

fig, ax = plt.subplots(1, 1, figsize=(15, 15))

# # Plot each district with a different color
# for gu in gdf['gu_name'].unique():
#     subset = gdf[gdf['gu_name'] == gu]
#     subset.plot(ax=ax, label=gu, markersize=5)
from matplotlib.colors import to_hex

# 색상 설정
unique_gu = gdf['gu_name'].unique()
colors = plt.cm.get_cmap('tab20', len(unique_gu)) 

for i, gu in enumerate(unique_gu):
    subset = gdf[gdf['gu_name'] == gu]
    color = to_hex(colors(i))  # 색상 지정
    subset.plot(ax=ax, label=gu, markersize=5, color=color)

plt.legend(title="Districts", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.show()
plt.clf()
---------------
# 강사님

# 구 이름 만들기
# 방법1 -> 강사님
# gu_name=list()
# for i in range(25):
#     gu_name.append(geo_seoul["features"][i]["properties"]["SIG_KOR_NM"])
# gu_name
# 
# #방법2
# gu_name = [geo_seoul["features"][i]["properties"]["SIG_KOR_NM"] for i in range(25))]
# gu_name

gu_name = list()
for i in range(len(geo_seoul["features"])):
  gu_name.append(geo_seoul["features"][i]['properties']["SIG_KOR_NM"])

def draw_seoul(num):
  gu_name = geo_seoul["features"][num]['properties']["SIG_KOR_NM"]
  coordinate_list = geo_seoul["features"][num]["geometry"]["coordinates"]
  coordinate_array = np.array(coordinate_list[0][0])
  x = coordinate_array[:,0]
  y = coordinate_array[:,1]
  
  return pd.DataFrame({"gu_name": gu_name, "x":x, "y":y})
draw_seoul(12)

result = pd.DataFrame({
  "gu_name": [],
  "x":[],
  "y":[]
}) 

for i in range(25):
  result = pd.concat([result,draw_seoul(i)], ignore_index = True)
result

sns.scatterplot(data=result, x="x", y="y", hue="gu_name", s=2, legend=False)
plt.show()
plt.clf()
----------
# 강남구랑 강남구가 아닌곳 다르게 표시
gu_name = list()
for i in range(len(geo_seoul["features"])):
  gu_name.append(geo_seoul["features"][i]['properties']["SIG_KOR_NM"])

def draw_seoul(num):
  gu_name = geo_seoul["features"][num]['properties']["SIG_KOR_NM"]
  coordinate_list = geo_seoul["features"][num]["geometry"]["coordinates"]
  coordinate_array = np.array(coordinate_list[0][0])
  x = coordinate_array[:,0]
  y = coordinate_array[:,1]
  
  return pd.DataFrame({"gu_name": gu_name, "x":x, "y":y})

result = pd.DataFrame({
  "gu_name": [],
  "x":[],
  "y":[]
}) 

for i in range(25):
  result = pd.concat([result,draw_seoul(i)], ignore_index = True)
result

# 그래프 그리기
#1
gangnam_df = result.assign(is_gangnam=np.where(result["gu_name"]=="강남구","강남","안강남"))
sns.scatterplot(data=gangnam_df, x="x", y="y", hue="is_gangnam",
                palette={"안강남" :"grey", "강남" : "red"},
                legend=False, s=2)
plt.show()
plt.clf()

#2
gangnam_df = result.assign(is_gangnam=np.where(result["gu_name"]=="강남구","강남","안강남"))
sns.scatterplot(data=gangnam_df, x="x", y="y", 
                palette=viridis,
                legend=False, s=2)
plt.show()
plt.clf()
-----------------------------------------------------
# 연습
for i in range(len(geo_seoul['features'])):
df_total = pd.DataFrame({
        "x": np.array(geo_seoul['features'][i]['geometry']['coordinates'][0][0])[:,0],
        "y": np.array(geo_seoul['features'][i]['geometry']['coordinates'][0][0])[:,1]
})

for i in range(len(geo_seoul['features'])):
    gu_name = geo_seoul['features'][i]['properties']['SIG_KOR_NM']
    for j in range(len(np.array(geo_seoul['features'][i]['geometry']['coordinates'][0][0])))
            df_gu = pd.DataFrame({
                        "gu_name" : gu_name 
                   })

df = pd.DataFrame(columns=['SIG_KOR_NM', 'x', 'y'])    
for i in range(25):
  df.loc[i] = [geo_seoul["features"][i]["properties"]["SIG_KOR_NM"], 
                geo_seoul["features"][i]["geometry"]["coordinates"][0][0][0][0],
                np.array(geo_seoul["features"][0]["geometry"]["coordinates"][0][0])
--------------------------------------------------------
#11-2와 11-1의 인구 정보 합치기
geo_seoul = json.load(open("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/SIG_Seoul.geojson", encoding="UTF-8"))
geo_seoul["features"][0]["properties"]
geo_seoul["features"][1]

df_pop = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1/data/Population_SIG.csv")
df_pop.head()
df_seoulpop=df_pop.iloc[1:26]
df_seoulpop["code"]=df_seoulpop["code"].astype(str)
df_seoulpop.info()

import folium
result["x"].mean()
result["y"].mean()

# p.304
map_sig=folium.Map(location=[37.55,126.97],zoom_start=12,
                 tiles="cartodbpositron")

folium.Choropleth(
    geo_data = geo_seoul,
    data=df_seoulpop,
    columns = ("code","pop"),
    key_on = "feature.properties.SIG_CD") \
     .add_to(map_sig)
map_sig.save("map_seoul.html")

# p.306
bins = list(df_pop["pop"].quantile([0, 0.2, 0.4, 0.6, 0.8, 1]))
bins

map_sig=folium.Map(location=[37.55,126.97],zoom_start=12,
                 tiles="cartodbpositron")

folium.Choropleth(
    geo_data = geo_seoul,
    data=df_seoulpop,
    columns = ("code","pop"),
    key_on = "feature.properties.SIG_CD",
    fill_color="YlGnBu",
    bins=bins)\
    .add_to(map_sig)
map_sig.save("map_seoul.html")

# 중앙에 점찍기
make_seouldf(0).iloc[:,1:3].mean()
folium.Marker([37.583744, 126.983800], popup="종로구").add_to(map_sig)
map_sig.save("map_seoul.html")

# 
