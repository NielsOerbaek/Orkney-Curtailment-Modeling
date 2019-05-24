import prepros as pp
import plotter
import descriptive as desc
import model as m
import metoffice as met

#plotter.buildModelGraph("2019-03-01", "2019-03-15")
#plotter.buildModelGraph("2019-01-01", "2019-01-15")
#plotter.buildModelGraph("2018-12-01", "2019-03-01")
#plotter.buildModelGraph("2019-02-12", "2019-03-01")
#plotter.buildWindsGraph("2019-02-15", "2019-03-01")
#plotter.buildEdayScatter("2019-01-13", "2019-01-30")
#plotter.buildEdayScatter("2019-02-15", "2019-03-01")

#pp.howClean(pp.getSingleDataframe(fromPickle=True))


#met.evaluateMetForecast()
#met.hitAccuracy()
#met.evaluateMetForecast(name="met-full-frame-all-clean", code=4)
#met.makeForecastAccTable(name="met-full-frame-all")
#met.makeForecastAccTable(name="met-full-frame-all-clean")
#met.getFullCombinedMetFrame()
plotter.buildAll()

#full = pp.getSingleDataframe(fromPickle=True, clean=True)
#api = pp.getSingleDataframe("2019-02-12", "2019-03-01", fromPickle=True)
#print(api.describe())

#
#desc.evaluateDataframe(full, full)
#desc.evaluateDataframe(api, api)


exit()

def makeEdayModel():
    df_train = pp.getEdayData()
    df_full = pp.getSingleDataframe(fromPickle=True)
    df_train = df_full.join(df_train,how="inner")
    m.train_and_save_simple(df_train[["Wind Mean (M/S)", "weekday", "hour"]].values, df_train[["Curtailment"]].values,kfold=False,filename="WT-FFNN-ERE")

makeEdayModel()
