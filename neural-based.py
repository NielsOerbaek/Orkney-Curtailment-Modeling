import prepros as pp
import model as m
import numpy as np
from keras.backend import clear_session

df_full = pp.getSingleDataframe("2018-12-01", "2019-03-01", fromPickle=True, clean=True)
#df_full = pp.getSingleDataframe("2019-02-12", "2019-03-01", fromPickle=True, clean=True)
df_eday = pp.getEdayData()
df = df_full.join(df_eday, how="inner")

wind_data = "Wind Mean (M/S)"#"speed"#

#df = pp.getSingleDataframe("2019-01-01", "2019-01-08", fromPickle=False)
#df = pp.getSingleDataframe("2019-03-01", "2019-03-21", fromPickle=False)


labels = ["Curtailment"]

print("------ EDAY WIND/TIME FFNN -------")
features = [wind_data,"hour","weekday"]
lengths = [1,24,7]
model = m.train_and_save_perceptron(df[features].values,df[labels].values,filename="wind-time-ffnn")
clear_session()
del model

print("------ GEN/DEM FFNN -------")
features = ["ANM Generation", "Non-ANM Generation", "Demand"]
model = m.train_and_save_simple(df[features].values,df[labels].values,filename="gen-dem-ffnn")
clear_session()
del model

print("------ FFNN With many things -------")
features = ["Generation", "Demand",wind_data,"hour","weekday"]
lengths = [1,24,7]
model = m.train_and_save_simple(df[features].values,df[labels].values,filename="wind-time-percep")
clear_session()
del model


print("------ GEN/DEM PERCEPTRON -------")
features = ["Generation", "Demand"]
model = m.train_and_save_perceptron(df[features].values,df[labels].values,filename="gen-dem-percep")
clear_session()
del model

print("------ WIND/TIME PERCEPTRON -------")
features = [wind_data,"hour","weekday"]
lengths = [1,24,7]
model = m.train_and_save_perceptron(df[features].values,df[labels].values,filename="wind-time-percep")
clear_session()
del model










#print("\n-----------------------\nEstimating wind speeds")
#df = pp.estimateWindSpeeds(df)
#m.train_and_save_simple(df[features].values,df[labels].values,filename="simple_neural")

#print("\n-----------------------\nChanging time data to one one-hots")
#df = pp.addTimeColsOneHot(df)
#vals = df[features].values
#input = np.zeros(shape=(vals.shape[0],sum(lengths)))
#for i,v in enumerate(vals):
#    v[0] = np.array([v[0]])
#    input[i] = np.concatenate(v)
#model = m.train_and_save_simple(input,df[labels].values,filename="simple_neural")
#
#print("\n-----------------------\nPseudo-Normalize Wind speeds")
#input[:,0] = input[:,0] / 20
#
#model = m.train_and_save_simple(input,df[labels].values,filename="simple_neural")
