import pandas as pd
from datetime import datetime
import pickle
import config

df = pd.read_csv(config.DATA_PATH+"eday/eday-winter-2018-2019.csv")

df["datetime"] = [datetime.strptime(df.iloc[i]["Date"] + " " + str(df.iloc[i]["Hour"]) + "-" + str(df.iloc[i]["Minute"]), '%m/%d/%Y %H-%M') for i in df.index]

df.index = df["datetime"]

df = df[["Wind Mean (M/S)","Wind Max (M/S)","Wind Min (M/S)","Power Mean (Kw)","Power Max (Kw)","Power Min (Kw)"]]

pickle.dump(df, open(config.DATA_PATH+"eday/eday-data.pickle", "wb"))
