import prepros as pp
import model as m
import numpy as np

df = pp.getSingleDataframe("2018-12-01", "2019-03-01", fromPickle=False)
#df = pp.getSingleDataframe("2019-02-11", "2019-03-01", fromPickle=True)
#df = pp.getSingleDataframe("2019-01-01", "2019-01-08", fromPickle=False)

#TODO: Change hour and weekday to one-hots.

features = ["speed","hour","weekday"]
lengths = [1,24,7]

labels = ["Curtailment"]

#TODO: Find out how well your estimation is doing.
print("\n-----------------------\nEvaluating model on base dataset")
m.train_and_save_simple(df[features].values,df[labels].values,filename="simple_neural")

print("\n-----------------------\nCleaning data")
df = pp.cleanData(df)
m.train_and_save_simple(df[features].values,df[labels].values,filename="simple_neural")

print("\n-----------------------\nEstimating wind speeds")
df = pp.estimateWindSpeeds(df)
m.train_and_save_simple(df[features].values,df[labels].values,filename="simple_neural")

print("\n-----------------------\nChanging time data to one one-hots")
df = pp.addTimeColsOneHot(df)
vals = df[features].values
input = np.zeros(shape=(vals.shape[0],sum(lengths)))
for i,v in enumerate(vals):
    v[0] = np.array([v[0]])
    input[i] = np.concatenate(v)

print(np.max(input[:,0]))

#TODO: Find out how well your estimation is doing.
model = m.train_and_save_simple(input,df[labels].values,filename="simple_neural")
