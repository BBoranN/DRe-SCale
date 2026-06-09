import json
from threading import Thread
import subprocess
from time import sleep
import numpy as np


FUNCTION_URL = "http://127.0.0.1:8000/function/matmul"

# Constant arrival rate in requests per second.
# For example, 1.0 means an average of 30 requests per 30-second window.
CONSTANT_LAMBDA = 0.5
TOTAL_TIME = 30


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

window_count = len(data)

# Setting the random seed for replication of results
seed = 29
np.random.seed(seed)

print(f"Starting constant workload generator targeting: {FUNCTION_URL}")
print(f"Using constant lambda: {CONSTANT_LAMBDA} requests/second")
print("Press Ctrl+C to stop...")

# running for longer period of time to simulate > 14 days
for _ in range(100):
    for _ in range(window_count):
        elapsed = 0.0

        try:
            while True:
                inter = np.random.exponential(scale=1 / CONSTANT_LAMBDA)
                elapsed += inter

                if elapsed > TOTAL_TIME:
                    sleep(max(0, TOTAL_TIME - (elapsed - inter)))
                    break

                sleep(inter)
                th = Thread(target=func, args=(1,))
                th.start()

        except Exception as e:
            print(f"Error: {e}")
