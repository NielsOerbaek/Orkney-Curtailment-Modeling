import time
import predict as p

starttime = time.time()
wait_secs = 10*60.0
 
while True:
    p.getAndStorePrediction()
    time.sleep(wait_secs - ((time.time() - starttime) % wait_secs))
