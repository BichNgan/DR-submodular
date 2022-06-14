import os
import csv

from tqdm import tqdm
import psutil
import numpy as np


def get_memory():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**3)\

def read_dataset(dataset, ignore_header=True, reindex=True, max_rating=5):
    """Return E and p(st) from dataset"""
    distinct_v = set()
    edges = list()
    num_lines = sum(1 for _ in open(dataset))
    # print(num_lines)
    with tqdm(total=num_lines, position=0,
              leave=False, desc="Reading data") as pbar:
        with open(dataset, 'r') as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                if ignore_header:
                    ignore_header=False
                    pbar.update(1)
                    continue
                u = int(row[0])
                v = int(row[1])
                w = float(row[2]) / max_rating
                distinct_v.add(u)
                distinct_v.add(v)
                edges.append([u,v,w])
                pbar.update(1)
    n = len(distinct_v)
    pst = np.zeros((n,n))

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
                pst[new_index[u], new_index[v]] = w
                pbar.update(1)
            return np.arange(n), pst
        for edge in edges:
            u, v, w = edge
            pst[u, v] = w
            pbar.update(1)
        return np.arange(n), pst

class OracleCounter:
    def __init__(self, f):
        self.count = 0
        self.f = f

    def __call__(self, *args, **kwds):
        self.count += 1
        return self.f(*args, **kwds)
