import prepros as pp
import plotter

#plotter.buildModelGraph("2019-03-01", "2019-03-15")
#plotter.buildModelGraph("2019-01-01", "2019-01-15")
#plotter.buildModelGraph("2018-12-01", "2019-03-01")
#plotter.buildWindsGraph("2019-02-15", "2019-03-01")
#plotter.buildEdayScatter("2019-01-13", "2019-01-30")
#plotter.buildEdayScatter("2019-02-15", "2019-03-01")

#pp.howClean(pp.getSingleDataframe(fromPickle=True))

plotter.buildAll()
