import random
import numpy as np
from time import time

# basic numpy matrix multiplication
def matmul(n):
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    start = time()
    C = np.matmul(A, B)
    latency = time() - start
    return latency

# openfaas event handler function
def handle(event, context):
    input = [10, 100, 1000]
    n = random.randint(0, 2)
    
    # Calculate latency
    result = matmul(input[n])
    
    # Return as a string to ensure valid HTTP response body
    return str(result)