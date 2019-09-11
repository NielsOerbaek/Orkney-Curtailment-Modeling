import time
import predict as p
import prepros as pp

def getAndStorePrediction():
    ts,f,dt = pp.getPredictionData()
    zp,rp,dt,td,m = p.makePrediction(ts,f,dt)
    return p.storePrediction(zp,rp,dt,td,m)

starttime = time.time()
wait_secs = 10*60.0

while True:
    getAndStorePrediction()
    time.sleep(wait_secs - ((time.time() - starttime) % wait_secs))
