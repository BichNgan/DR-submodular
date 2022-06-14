from concurrent.futures import ProcessPoolExecutor

import numpy as np

from tools import OracleCounter

@OracleCounter
def monotone_reduction(n):
    reduce_vector = np.random.uniform(low=1, high=100, size=n)
    def function(x):
        return np.dot(x,reduce_vector) 
    return function

@OracleCounter
def budget_allocation(n, pst=None, workers=1):
    if pst is None:
        pst = np.random.uniform(size=(n,n))
        for index in range(len(pst)):
            thres = np.random.uniform(0.5, 0.9)
            pst[index, pst[index] < thres] = 0
            pst[index, index] = 0
    wst = 1 - pst
    t_wst = wst.transpose()
    pool = ProcessPoolExecutor(max_workers=workers)
    
    def function(x):
        _x = np.array(x)
        product_of_power = lambda wi: 1 - np.prod(pow(wi, _x))
        sources = _x > 0
        targets = np.sum(pst[sources], axis=0) > 0
        return np.sum(list(pool.map(product_of_power, t_wst[targets])))
    
    return function
