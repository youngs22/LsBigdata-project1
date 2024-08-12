# import os
# cwd = os.getcwd()
# cwd
# os.chdir('c:/Users/USER/Documents/LS빅데이터스쿨/LsBigdata-project1')
# 주석처리 ctrl + /

# p 70
import pandas as pd
import numpy as np
import plotly.express as px
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()

fig = px.scatter(
    penguins,
    x= "bill_length_mm",
    y= "bill_depth_mm",
    color="species",
    )
fig.update_layout(
    
    title = { "text": "<span style = 'color:blue;font-weight:bold'> 팔머펭귄 </span>", \
        "x" : 0.5, "xanchor": "right"}
) # 팔머펭귄 굵게 만듬 & 색 바꿈

# CSS 문법  
# <span> 
# <span style = "font-weight:bold"> ... </span>
# <span> ... </span>
# <span> ... </span>
# </span>


