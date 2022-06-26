import os
import csv

from tqdm import tqdm
import psutil
import numpy as np


def get_memory():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**3)

def read_dataset(dataset, reindex=True, max_rating=5, delimiter=','):
    """Return E and p(st) from dataset"""
    distinct_v = set()
    edges = list()
    num_lines = sum(1 for _ in open(dataset))
    with tqdm(total=num_lines, position=0,
              leave=False, desc="Reading data") as pbar:
        with open(dataset, 'r') as file:
            reader = csv.reader(file, delimiter=delimiter)
            for row in reader:
                try:
                    u = int(row[0])
                    v = int(row[1])
                    w = float(row[2]) / max_rating
                    distinct_v.add(u)
                    distinct_v.add(v)
                    edges.append([u,v,w])
                except:
                    pass
                pbar.update(1)
    n = len(distinct_v)
    pst = np.zeros((n,n))
    is_source = np.full(n, False)

    with tqdm(total = n + len(edges),
              position=0, leave=False, 
              desc="Preparing data") as pbar:
        distinct_v = list(distinct_v)
        distinct_v.sort()
        if reindex:
            new_index = dict()
            for i, v in enumerate(distinct_v):
                new_index[v] = i
                pbar.update(1)
            for edge in edges:
                u, v, w = edge
                is_source[new_index[u]] = True
                pst[new_index[u], new_index[v]] = w
                pbar.update(1)
            return np.arange(n), pst, is_source
        for edge in edges:
            u, v, w = edge
            is_source[u] = True
            pst[u, v] = w
            pbar.update(1)
    return np.arange(n), pst, is_source

class OracleCounter:
    def __init__(self, f):
        self.count = 0
        self.f = f

    def __call__(self, *args, **kwds):
        self.count += 1
        return self.f(*args, **kwds)

    def reset(self):
        self.count = 0
