import numpy as np

from tools import OracleCounter

x: np.ndarray 

def product_of_power(wi):
    return 1 - np.prod(np.power(wi, x))

def monotone_reduction(n):
    reduce_vector = np.random.uniform(low=1, high=100, size=n)
    reduce_vector /= np.sum(reduce_vector)
    def function(x):
        return np.dot(x,reduce_vector)*n 
    return OracleCounter(function)


def budget_allocation(n, pst=None):
    if pst is None:
        pst = np.random.uniform(size=(n,n))
        for index in range(n):
            thres = np.random.uniform(0.5, 0.9)
            pst[index, pst[index] < thres] = 0
            pst[index, index] = 0
    wst_t = (1 - pst).transpose()
    def function(_x):
        nonlocal wst_t, pst
        global x
        x = _x
        sources = x > 0
        targets = np.dot(np.ones(len(pst[sources])), pst[sources]) > 0 # this is where it's stuck!!!!!
        return sum(map(product_of_power, wst_t[targets])) 
    return OracleCounter(function)
