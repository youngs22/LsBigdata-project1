import pandas as pd
import numpy as np

df = pd.read_clipboard()
df

df=df.sort_values("성별", ignore_index=True)
df_f=df.loc[:12]
df_m=df.loc[13:].reset_index(drop=False)

np.random.seed(20240827)
team1_f=np.random.choice(df_f["이름"], 6, replace=False)
team1_m=np.random.choice(df_m["이름"], 6, replace=False)

  

