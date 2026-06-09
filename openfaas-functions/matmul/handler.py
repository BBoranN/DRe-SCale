import random
import numpy as np
from time import time

# basic numpy matrix multiplication
# def matmul(n):
#     A = np.random.rand(n, n)
#     B = np.random.rand(n, n)

#     start = time()
#     C = np.matmul(A, B)
#     latency = time() - start
#     return latency

# # openfaas event handler function
# def handle(event, context):
#     # input = [10, 100, 1000]
#     # n = random.randint(0, 2)
#     n = 4000
    
#     # Calculate latency
#     result = matmul(n)
    
#     # Return as a string to ensure valid HTTP response body
#     return str(result)


def matmul_workload(n, iterations):
    # n=800 consumes only about 15MB of RAM total. Extremely safe!
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    start = time()
    
    # We loop the multiplication to rack up CPU time 
    # without allocating any new memory.
    for _ in range(iterations):
        C = np.matmul(A, B)
        
    latency = time() - start
    return latency

# openfaas event handler function
def handle(event, context):
    n = 1400 
    
    # TWEAK THIS NUMBER: 
    # This controls how many times the matrices are multiplied.
    # On a 150m CPU, 10 iterations should take roughly 1 to 2 seconds.
    iterations = 1
    
    # Calculate latency
    result = matmul_workload(n, iterations)
    
    # Return as a string to ensure valid HTTP response body
    return str(result)