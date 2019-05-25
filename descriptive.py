import prepros as pp
from datetime import datetime
import numpy as np
import model as m
import config

from scipy.interpolate import interp1d

def makeDescriptiveDataset(start, stop, clean=True, eday=False):
    df= pp.getSingleDataframe(start, stop, fromPickle=True)
    if eday:
        df_eday = pp.getEdayData()
        df = df.join(df_eday, how="inner")
    df = pp.addReducedCol(df)
    df.dropna(inplace=True) # Remove NaN entries.

    if clean:
        print("Cleaning data...")
        df = pp.cleanData(df)
        df = pp.addReducedCol(df, clean=True)
        df = pp.removeGlitches(df)

    return df

def simpleModel(dem, gen): return gen > dem + 20
def neverCurtail(dem, gen): return 0
def alwaysCurtail(dem, gen): return 1
def simpleModelK(dem, gen,k): return gen > dem + k

models = [  ("Simple Model   ", simpleModel),
            ("Never Curtail  ", neverCurtail),
            ("Always Curtail ", alwaysCurtail)]

D = [[ 15.635, 17.215, 18.155, 17.78, 18.265, 18.495, 19.06, 21.245, 22.145, 22.59, 21.51, 20.59, 22.705, 22.4, 21.42, 21.13, 21.955, 22.39, 23.065, 21.27, 21.89, 22.045, 18.1, 17.425 ],
    [ 15.585, 17.67, 17.635, 17.46, 18.14, 18.48, 18.905, 20.45, 21.44, 22.115, 21.235, 21.04, 23.13, 22.44, 20.94, 21.43, 21.76, 21.88, 21.665, 21.18, 21.7, 21.16, 17.915, 17.045 ],
    [ 15.725, 17.545, 17.805, 17.14, 17.865, 18.54, 18.66, 20.475, 21.26, 21.08, 20.18, 20.27, 22.42, 21.775, 19.715, 20.21, 20.965, 21.61, 22.43, 20.905, 21.395, 21.405, 17.945, 17.055 ],
    [ 15.72, 17.7, 17.665, 17.36, 17.57, 18.025, 18.2, 19.845, 20.325, 21.28, 20.25, 20.285, 22.105, 21.38, 18.93, 19.94, 20.65, 21.02, 22.19, 20.5, 21.045, 21.1, 17.57, 16.805 ],
    [ 15.625, 17.635, 17.41, 16.99, 17.765, 17.86, 18.425, 20.18, 20.305, 20.955, 20.32, 19.9, 21.84, 21.145, 19.31, 20.125, 21.62, 21.91, 22.165, 21.15, 21.14, 21.145, 17.725, 17.2 ],
    [ 16.075, 18.33, 18.02, 17.655, 17.76, 18.23, 18.745, 19.27, 18.815, 18.885, 19.135, 19.96, 22.13, 21.52, 19.33, 20.12, 20.795, 21.28, 22.13, 20.7, 21.06, 21.02, 18.12, 17.78 ],
    [ 16.44, 18.485, 18.905, 18.235, 18.615, 18.73, 18.545, 19.455, 18.3, 18.32, 18.945, 20.005, 21.525, 20.94, 19.415, 19.655, 20.79, 21.345, 21.925, 20.11, 20.875, 21.59, 17.94, 17.5 ]]


windToGenPoints = pp.getSingleDataframe("2019-02-12","2019-03-01",fromPickle=True)[["speed", "Generation"]].round({"speed": 0}).groupby("speed").median().values[:12,0]
windToGenInterp = interp1d(range(0,len(windToGenPoints)), windToGenPoints, bounds_error=False, fill_value=30)
edayToGenPoints = [1,1,2,2,3,4,7,11,16,21,24,29,31,32,33,34,34,34,34,35,34,35,30,31,27.5,30.5,24,20,12,9,9,9,6,6,9.5,4.5]
edayToGenInterp = interp1d(range(0,len(edayToGenPoints)), edayToGenPoints, bounds_error=False, fill_value=30)

def correlationModelK(w,d,h,k): return windToGenLinear(w) > (D[d-1][h-1] + k)
def correlationModelKPoly(w,d,h,k): return windToGenPolyFull(w) > (D[int(d)-1][int(h)-1] + k)
def correlationModelKCurve(w,d,h,k): return windToGenInterp(w) > (D[int(d)-1][int(h)-1] + k)
def correlationModelKCurveEday(w,d,h,k): return edayToGenInterp(w) > (D[int(d)-1][int(h)-1] + k)

def evaluateModels(start, stop, clean=True, onlySCk=False):
    df = makeDescriptiveDataset(start, stop, clean=clean)
    df_api = df.loc[datetime.strptime("2019-02-12", '%Y-%m-%d'):datetime.strptime("2019-03-01", '%Y-%m-%d')]
    samples = len(df.values)
    api_samples = len(df_api.values)
    wind_data = "speed" #"Wind Mean (M/S)"
    wind_index = np.where(df.columns.values==wind_data)[0]
    weekday_index = np.where(df.columns.values=="weekday")[0]
    hour_index = np.where(df.columns.values=="hour")[0]

    index = list(range(-20, 21))
    values = []
    names = []
    styles = []

    vs = []
    print("$GD_k$")
    for k in index:
        hits = 0
        print(".",end="", flush=True)
        for v in df.values:
            if simpleModelK(v[0],v[1],k) == v[-1]: hits += 1
        vs.append(round(hits/samples*100,2))
    values.append(vs)
    names.append("$GD_k$")
    styles.append("b-")
    print(" - {}% at k = {}".format(max(vs), index[vs.index(max(vs))]))

    if onlySCk: return [index, values, names, styles]

    vs = []
    print("$WT_k$ with interpolated power curve")
    for k in index:
        hits = 0
        print(".",end="", flush=True)
        for v in df.values:
            if correlationModelKCurve(v[wind_index],int(v[weekday_index]),int(v[hour_index]),k) == v[-1]: hits += 1
        vs.append(round(hits/samples*100,2))
    values.append(vs)
    names.append("$WT_k$")
    styles.append("r-")
    print(" - {}% at k = {}".format(max(vs), index[vs.index(max(vs))]))

    vs = []
    print("$GD_k$ - API")
    for k in index:
        hits = 0
        print(".",end="", flush=True)
        for v in df_api.values:
            if simpleModelK(v[0],v[1],k) == v[-1]: hits += 1
        vs.append(round(hits/api_samples*100,2))
    values.append(vs)
    names.append("$GD_k$ - API")
    styles.append("b--")
    print(" - {}% at k = {}".format(max(vs), index[vs.index(max(vs))]))

    vs = []
    print("$WT_k$ - API -  with interpolated power curve")
    for k in index:
        hits = 0
        print(".",end="", flush=True)
        for v in df_api.values:
            if correlationModelKCurve(v[wind_index],int(v[weekday_index]),int(v[hour_index]),k) == v[-1]: hits += 1
        vs.append(round(hits/api_samples*100,2))
    values.append(vs)
    names.append("$WT_k$ - API")
    styles.append("r--")
    print(" - {}% at k = {}".format(max(vs), index[vs.index(max(vs))]))

    return [index, values, names, styles]

def evaluateDataframe(df_train,df_predict):
    wind_data = "speed"

    wtnn = m.train_and_save_simple(df_train[[wind_data, "weekday", "hour"]].values, df_train[["Curtailment"]].values,kfold=False)
    gdnn = m.train_and_save_simple(df_train[["Demand", "Generation"]].values, df_train[["Curtailment"]].values,kfold=False)

    def actual(row): return row["Curtailment"]
    def gd(row): return simpleModelK(row["Demand"],row["Generation"],7)
    def wt(row): return correlationModelKCurve(row[wind_data],row["weekday"],row["hour"],6)
    def nn_gen_dem(row): return  round(gdnn.predict([[row[["Demand", "Generation"]].values]])[0][0])
    def nn_wind_time(row): return  round(wtnn.predict([[row[[wind_data, "weekday","hour"]].values]])[0][0])

    models = ["Actual", "$GD_7$", "GD FFNN", "$WT_6$","WT FFNN"]
    model_functions = [actual, gd, nn_gen_dem, wt, nn_wind_time]
    models_accs = np.zeros(shape=(len(models),))

    # Generate x,y data for the mesh plot
    print("Generating predictions for", df_predict.values.shape[0],"samples on", len(models), "models.")
    accs = np.zeros(shape=(len(models),len(df_predict.index)))
    for i, row in enumerate(df_predict.iterrows()):
        if i % 100 == 0: print(".", end="", flush=True)
        for j, model in enumerate(models):
            accs[j,i] = model_functions[j](row[1])
            if accs[j,i] == accs[0,i]: models_accs[j] += 1
    print()

    models_accs = models_accs / df_predict.values.shape[0]

    for i, model in enumerate(models):
        print(model, ":", round(models_accs[i]*100,2), "%")

    return models, accs
