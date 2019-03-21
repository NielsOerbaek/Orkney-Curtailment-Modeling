import prepros as pp

df = pp.getSingleDataframe("2019-02-11", "2019-03-01")

dfc = pp.cleanData(df)


print(dfc)
