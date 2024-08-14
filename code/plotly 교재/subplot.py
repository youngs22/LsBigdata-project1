import pandas as pd
import numpy as np
import plotly.express as px
from palmerpenguins import load_penguins

penguins = load_penguins()
# penguins.head()

fig = px.scatter(
    penguins,
    x="bill_length_mm",
    y="bill_depth_mm",
    color="species",
    # trendline="ols" # p.134
)
fig.show()

from plotly.subplots import make_subplots

fig_subplot = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Adeli","Gentoo","Chinstrap")
)

fig_subplot.add_trace(
    {
    "type":"scatter",
    "mode":"markers",
    "x":penguins.query("species=='Adelie'")["bill_length_mm"],
    "y":penguins.query("species=='Adelie'")["bill_depth_mm"],
    "name":"Adeli"
    },
row=1, col=1
)

fig_subplot.add_trace(
    {
    "type":"scatter",
    "mode":"markers",
    "x":penguins.query("species=='Gentoo'")["bill_length_mm"],
    "y":penguins.query("species=='Gentoo'")["bill_depth_mm"],
    "name":"Gentoo"
    },
row=1, col=2
)

fig_subplot.add_trace(
    {
    "type":"scatter",
    "mode":"markers",
    "x":penguins.query("species=='Chinstrap'")["bill_length_mm"],
    "y":penguins.query("species=='Chinstrap'")["bill_depth_mm"],
    "name":"Chinstrap"
    },
row=1, col=3
)

fig_subplot.update_layout(
    title=dict(text="펭귄종별 부리 길이 vs 깊이",
               x=0.5)
)

