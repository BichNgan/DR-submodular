import subprocess

import numpy as np

from algorithms import Algorithm2, Algorithm4, ThresholdGreedy

class Configuration:
    def __init__(self):
        self.algs = {
                'alg2': Algorithm2,
                'alg4': Algorithm4,
                'tg': ThresholdGreedy
                }
        self.data_dir = 'data'
        self.datasets = {
                'wiki': {
                    'dir': self.data_dir + '/out.wikilens-ratings',
                    'delimiter': '\t',
                    'output': 'wikilens.csv',
                    'max_weight': 5
                    },
                'movie': {
                    'dir': self.data_dir + '/out.movielens-1m',
                    'delimiter': ' ',
                    'output': 'movielens.csv',
                    'max_weight': 5
                    },
                'enron': {
                    'dir': self.data_dir + '/out.bag-enron',
                    'delimiter': '\t',
                    'output': 'bag_enron.csv',
                    'max_weight': 403
                    }
                }
        self.log_dir = 'log'
        self.output_dir = 'output'
        self.k_values = np.array([200, 400, 600, 800, 1000])
        self.b_max = 10
        self.epsilon = 0.1
        self.init_dir()

    def get_alg_ids(self):
        return list(self.algs.keys())

    def mkdir(self, dir):
        subprocess.run(f'mkdir {dir}', shell=True)

    def init_dir(self):
        result = subprocess.check_output('ls', shell=True, universal_newlines=True)
        result = result.split('\n')
        if self.data_dir not in result:
            self.mkdir(self.data_dir) 
        if self.log_dir not in result:
            self.mkdir(self.log_dir)
        if self.output_dir not in result:
            self.mkdir(self.output_dir)
