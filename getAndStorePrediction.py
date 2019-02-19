import time
import predict as p
import prepros as pp

starttime = time.time()
wait_secs = 10*60.0

while True:
    ts,f,dt = pp.getPredictionData()
    zp,rp,dt,td = p.makePrediction(ts,f,dt)
    p.storePrediction(zp,rp,dt,td)
    time.sleep(wait_secs - ((time.time() - starttime) % wait_secs))
