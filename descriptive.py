import prepros as pp
from datetime import datetime
import numpy as np

def makeDescriptiveDataset(start, stop):
    df = pp.getSingleDataframe(start, stop)
    df.dropna(inplace=True) # Remove NaN entries.

    print("Cleaning data...")
    df = pp.cleanData(df)
    df = pp.addReducedCol(df)

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

poly_coefs_11 = [-0.0703,1.048,-0.891,2.2]
poly_coefs_full = [-0.0262,0.3693,2.09,-1.626]

windToGenLinear = lambda w: max(3.501*w + 2.915,0)
windToGenPolyFull = np.poly1d(poly_coefs_full)
windToGenPoly11 = np.poly1d(poly_coefs_11)
def correlationModelK(w,d,h,k): return windToGenLinear(w) > (D[d-1][h-1] + k)
def correlationModelKPoly(w,d,h,k): return windToGenPolyFull(w) > (D[d-1][h-1] + k)

def evaluateModels(start, stop):
    df = makeDescriptiveDataset(start, stop)
    samples = len(df.values)
    print(samples)
    print("--- Model Accuracies:")
    for m in models:
        hits = 0
        for v in df.values:
            if m[1](v[0],v[1]) == v[-1]: hits += 1
        print(m[0],"&", str(round(hits/samples*100,2))+"%", "\\\\\\hlline")

    print("--- Accuracies for SC_k for different values of k")
    for k in range(-20, 21):
        hits = 0
        for v in df.values:
            if simpleModelK(v[0],v[1],k) == v[-1]: hits += 1
        print("(", k,",",round(hits/samples*100,2),")")

    print("--- Accuracies for Linear CC_k for different values of k")
    for k in range(-20, 21):
        hits = 0
        for v in df.values:
            if correlationModelK(v[2],int(v[9]),int(v[6]),k) == v[-1]: hits += 1
        print("(", k,",",round(hits/samples*100,2),")")
    print("--- Accuracies for Poly CC_k for different values of k")
    for k in range(-20, 21):
        hits = 0
        for v in df.values:
            if correlationModelKPoly(v[2],int(v[9]),int(v[6]),k) == v[-1]: hits += 1
        print("(", k,",",round(hits/samples*100,2),")")


#evaluateModels("2019-02-18", "2019-02-25")
#evaluateModels("2019-02-11", "2019-03-01")
evaluateModels("2018-12-01", "2019-03-01")
