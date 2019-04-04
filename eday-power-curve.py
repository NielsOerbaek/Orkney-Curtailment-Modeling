import numpy as np
import prepros as pp

df = pp.getEdayData()
df_full = pp.getSingleDataframe(fromPickle=True)

#df = df_full.join(df,how="inner")
#df = df[df["Curtailment"] == 0]
df = df[((df["Wind Mean (M/S)"] < 4) | (df["Wind Mean (M/S)"] > 22) | (df["Power Mean (Kw)"] > 10)) & ((df["Wind Mean (M/S)"] < 12) | (df["Wind Mean (M/S)"] > 22) | (df["Power Mean (Kw)"] > 700))]

df = df[["Wind Mean (M/S)", "Power Mean (Kw)"]].round().groupby("Wind Mean (M/S)").median()


print(df)

for r in df.values:
    print(r[0])
