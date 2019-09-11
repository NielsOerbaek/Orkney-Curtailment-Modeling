import predict as p
import prepros as pp

def getAndStorePrediction():
    ts,f,dt = pp.getPredictionData()
    zp,rp,dt,td,m = p.makePrediction(ts,f,dt)
    return p.storePrediction(zp,rp,dt,td,m)

getAndStorePrediction()
