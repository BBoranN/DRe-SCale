import json
from threading import Thread
import subprocess
from time import sleep
import numpy as np
import math


FUNCTION_URL = "http://127.0.0.1:8080/function/matmul" 

def func(a=1):
    # We insert the URL directly into the command
    # -n 1 -c 1 means "1 request, 1 concurrent user"
    command = f"hey -n {a} -c {a} {FUNCTION_URL}"
    
    # Run slightly quietly so we don't spam the terminal too much
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL) 

filename = './invokation_data.json' 

try:
    with open(filename) as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: Could not find {filename}. Did you download it from the repo?")
    exit(1)

d = [int(i) for i in data]

# Setting the random seed for replication of results (as per paper)
seed = 29
np.random.seed(seed)

print(f"Starting workload generator targeting: {FUNCTION_URL}")
print("Press Ctrl+C to stop...")

# running for longer period of time to simulate > 14 days
for _ in range(20):
    for i in d: # number of requests per time interval i.e. 30 seconds
        total_time = 30
        try:
            if i == 0:
                # No requests this interval
                sleep(30)
                continue

            # average inter arrival time in seconds (lambda)
            average_inter_arrival_time = total_time/i
            
            # generate inter arrival times
            times = np.random.poisson(lam=average_inter_arrival_time, size=i)
            
            # for every request, generate and wait
            for k in range(0, i): 
                inter = times[k]
                th = Thread(target=func, args=(1,))
                th.start()
                if inter > 0:
                    sleep(inter)
                    
        except Exception as e:
            print(f"Error: {e}")
            
        # Ensure we stay synced to the 30s window (simple correction)
        # (The original code logic for 'times' sum was a bit specific, 
        # usually simpler is better for testing, but let's stick to the loop structure)