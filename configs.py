import subprocess

import numpy as np

data_dir = 'data'
datasets = ['out.wikilens-ratings','out.movielens-1m']
delimiters = ['\t', ' ']
log_dir = 'log'
output_dir = 'output'
k_values = np.array([50, 100, 500, 1000, 2000])
b_values = (k_values * 0.01).astype(int)
b_values[b_values <= 0] = 1
epsilon = 0.1
max_workers = 2 

def init_config():
    result = subprocess.check_output('ls', shell=True, universal_newlines=True)
    result = result.split('\n')
    if data_dir not in result:
        subprocess.run(f'mkdir {data_dir}', shell=True)
    if log_dir not in result:
        subprocess.run(f'mkdir {log_dir}', shell=True)
    if output_dir not in result:
        subprocess.run(f'mkdir {output_dir}', shell=True)
