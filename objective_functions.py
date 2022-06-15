from concurrent.futures import ProcessPoolExecutor

import numpy as np

from tools import OracleCounter

power_x = 0
pool: ProcessPoolExecutor 

def init_pool(max_workers):
    global pool
    pool = ProcessPoolExecutor(max_workers=max_workers)

def product_of_power(wi):
    return 1 - np.prod(pow(wi, power_x))

def monotone_reduction(n):
    reduce_vector = np.random.uniform(low=1, high=100, size=n)
    def function(x):
        return np.dot(x,reduce_vector) 
    return OracleCounter(function)


def budget_allocation(n, pst=None, workers=1):
    if pst is None:
        pst = np.random.uniform(size=(n,n))
        for index in range(len(pst)):
            thres = np.random.uniform(0.5, 0.9)
            pst[index, pst[index] < thres] = 0
            pst[index, index] = 0
    wst = 1 - pst
    t_wst = wst.transpose()
    init_pool(workers)
    def function(x):
        global power_x, pool
        _x = np.array(x)
        power_x = _x
        sources = _x > 0
        targets = np.sum(pst[sources], axis=0) > 0
        return np.sum(list(pool.map(product_of_power, t_wst[targets])))
    
    return OracleCounter(function)
