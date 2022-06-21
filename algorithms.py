from loguru import logger
from tqdm import tqdm
import numpy as np

from tools import get_memory

def log_base_n(n, x):
    return np.log(x) / np.log(n)

class Algorithm2:
    def __init__(self, e_arr, b_arr, f, k, epsilon):
        self.e_arr = e_arr
        self.b_arr = b_arr
        self.f = f
        self.k = k
        self.epsilon = epsilon
        self.memory = 0

    def __generate_o(self, m):
        o_min = int(np.ceil(log_base_n(1 + self.epsilon, m)))
        o_max = int(np.floor(log_base_n(1 + self.epsilon, 2*self.k*m)))
        o_power = np.arange(o_min, o_max+1)
        o_base = np.full(len(o_power), 1 + self.epsilon)
        o_arr = list(set(np.ceil(np.power(o_base, o_power)).astype(int)))
        o_arr.sort()
        return o_arr

    def __generate_i(self, be):
        i_min = int(np.ceil(log_base_n((1 + self.epsilon), 1/be)))
        i_max = int(np.floor(log_base_n((1 + self.epsilon), 1)))
        i_power = np.arange(i_min, i_max+1)
        i_base = np.full(len(i_power), 1 + self.epsilon)
        i_arr = set(np.ceil(np.power(i_base, i_power)*be).astype(int))
        return list(i_arr)

    def __find_ke(self, xv, xe, i_arr, v):
        fx = self.f(xv)
        delta_f = np.array([(self.f(xv + xe*i) - fx) for i in i_arr])
        i_np = np.array(i_arr)
        evaluation = i_np * v / (2 * self.k)
        try:
          result = np.min(i_np[delta_f < evaluation])
          return result
        except:
          return 0

    def __binary_search(self, x_arr, xe, i_arr, v):
        l = 0
        r = len(i_arr) - 1
        xv = x_arr[v]
        fx = self.f(xv)
        if self.f(xv + i_arr[0] * xe) - fx < i_arr[0] * v / (2 * self.k):
            return i_arr[0]
        while r > l:
            m = (l + r) // 2
            df = self.f(xv + i_arr[m] * xe) - fx
            threshold = i_arr[m] * v / (2 * self.k)
            if df >= threshold:
                l = m + 1
                continue
            r = m - 1
        return i_arr[l]

    @logger.catch
    def run(self):
        x_arr = dict()
        m = 0
        n = len(self.e_arr)
        with tqdm(total=n, leave=False, desc="Algorithm 2") as pbar:
            for e in self.e_arr:
                xe = np.full(n, 0)
                xe[e] = 1
                m = max(self.f(xe), m)
                o_arr = self.__generate_o(m)
                i_arr = self.__generate_i(self.b_arr[e])
                for v in o_arr:
                    if v not in x_arr.keys():
                        x_arr[v] = np.full(n, 0)
                    ke = self.__binary_search(x_arr, xe, i_arr, v)
                    knew = min(ke, self.k - np.sum(x_arr[v]))
                    if knew > 0:
                        x_arr[v] += knew*xe
                    else:
                        break
                pbar.update(1)
        x_list = [x_arr[key] for key in x_arr.keys()]
        fx_list = [self.f(x) for x in x_list]
        self.memory = get_memory()
        return x_list[np.argmax(fx_list)]

class Algorithm3:
    def __init__(self, e_arr, b_arr, f, k, epsilon):
        self.e_arr = e_arr
        self.b_arr = b_arr
        self.f = f
        self.k = k
        self.epsilon = epsilon
        self.memory = 0
    

    def __generate_i(self, be):
        i_min = int(np.ceil(log_base_n((1 + self.epsilon), 1/be)))
        i_max = int(np.floor(log_base_n((1 + self.epsilon), 1)))
        i_power = np.arange(i_min, i_max+1)
        i_base = np.full(len(i_power), 1 + self.epsilon)
        i_arr = set(np.ceil(np.power(i_base, i_power)*be).astype(int))
        return list(i_arr)

    def __find_ke(self, x, xe, i_arr):
        fx = self.f(x)
        if fx == 0 or np.sum(x) < self.k:
            return np.min(i_arr)
        i_np = np.array(i_arr)
        f_left = np.array([(self.f(x + xe*i) - fx) for i in i_arr])
        f_right = np.array([i * fx/self.k for i in i_arr])
        try:
            return np.min(i_np[f_left < f_right])
        except:
            return 0
    
    @logger.catch
    def run(self):
        n = len(self.e_arr)
        x = np.full(n, 0)
        with tqdm(total=n, leave=False, desc="Algorithm 3") as pbar:
            for e in self.e_arr:
                be = self.b_arr[e]
                xe = np.full(n, 0)
                xe[e] = 1
                i_arr = self.__generate_i(be)
                ke = self.__find_ke(x, xe, i_arr)
                x += ke * xe
                pbar.update(1)
        x_new = np.full(n, 0)
        has_at_least_one = False
        x_sum = 0
        for index in reversed(range(n)):
            if x[index] > 0 and not has_at_least_one:
                x_new[index] = x[index]
                has_at_least_one = True
                x_sum += x[index]
                continue
            if x_sum + x[index] > self.k:
                break
            x_new[index] = x[index]
            x_sum += x[index]
        self.memory = get_memory()
        return x_new

class Algorithm4:
    def __init__(self, e_arr, b_arr, f, k, epsilon):
        self.e_arr = e_arr
        self.b_arr = b_arr
        self.f = f
        self.k = k
        self.epsilon = epsilon
        self.memory = 0

    def __generate_i(self, be):
        i_min = int(np.ceil(log_base_n((1 + self.epsilon), 1/be)))
        i_max = int(np.floor(log_base_n((1 + self.epsilon), 1)))
        i_power = np.arange(i_min, i_max+1)
        i_base = np.full(len(i_power), 1 + self.epsilon)
        i_arr = set(np.ceil(np.power(i_base, i_power)*be).astype(int))
        return list(i_arr)

    def __find_ke(self, x, xe, i_arr, theta):
        fx = self.f(x)
        delta_f = np.array([(self.f(x + xe*i) - fx) for i in i_arr])
        i_np = np.array(i_arr)
        try:
            result = np.min(i_np[delta_f < theta])
            return result
        except:
            return 0

    def __binary_search(self, x, xe, i_arr, theta):
        l = 0
        r = len(i_arr) - 1
        fx = self.f(x)
        if self.f(x + i_arr[0] * xe) - fx < theta:
            return i_arr[0]
        while r > l:
            m = (l + r) // 2
            df = self.f(x + i_arr[m] * xe) - fx
            if df >= theta:
                r = m - 1
                continue
            break
        try:
            return min(i_arr[l:r+1])
        except:
            return 0
    
    @logger.catch
    def run(self):
        algorithm3 = Algorithm3(self.e_arr, self.b_arr, self.f, self.k, self.epsilon)
        gamma = self.f(algorithm3.run())
        theta = 2 * (2 - self.epsilon) * gamma /((1 - self.epsilon) * self.k)
        n = len(self.e_arr)
        x = np.full(n, 0)
        xe_dict = {}
        exit_threshold = (1 - self.epsilon) * gamma / (4 * self.k) 
        estimated_iterations = np.ceil(log_base_n(1 + self.epsilon, exit_threshold/theta))
        logger.info(f'Estimated iter: {estimated_iterations}, exit: {exit_threshold}, gamma: {gamma}')
        with tqdm(total=estimated_iterations, leave=False, desc="Algorithm 4") as pbar:
            while theta >= (1 - self.epsilon) * gamma / (4 * self.k):
                for e in self.e_arr:
                    if e not in xe_dict:
                        xe_dict[e] = np.full(n, 0)
                        xe_dict[e][e] = 1
                    be = self.b_arr[e]
                    i_arr = self.__generate_i(be)
                    ke = self.__binary_search(x, xe_dict[e], i_arr, theta)
                    k_new = min(ke, self.k - np.sum(x))
                    if k_new != 0:
                        x += xe_dict[e] * k_new
                    else:
                        break
                pbar.update(1)
                theta = (1 - self.epsilon) * theta
        self.memory = get_memory()
        return x

class ThresholdGreedy:
    def __init__(self, e_arr, b_arr, f, k, epsilon):
        self.e_arr = e_arr
        self.b_arr = b_arr
        self.f = f
        self.k = k
        self.epsilon = epsilon
        self.memory = 0

    def __binary_search(self, x, e, tau):
        l = 1
        r = min(self.b_arr[e] - x[e], self.k - sum(x))
        xe = np.zeros(len(x))
        xe[e] = 1
        fx = self.f(x)
        if self.f(x + r*xe) - fx >= tau:
            return r
        if self.f(x + xe) - fx < tau:
            return 0
        while r > l + 1:
            m = (l + r) // 2
            if self.f(x + m*xe) - fx >= tau:
                l = m
                continue
            r = m
        return l

    @logger.catch
    def run(self):
        n = len(self.e_arr)

        def create_xe(e):
            xe = np.zeros(n)
            xe[e] = 1
            return xe

        x = np.zeros(n)
        ls_xe = [create_xe(e) for e in self.e_arr]
        d = max([self.f(xe) for xe in ls_xe])
        tau = d
        threshold = self.epsilon / self.k * d
        with tqdm(total = self.k, leave=False, desc="Threshold Greedy") as pbar:
            while tau >= threshold:
                for e in self.e_arr:
                    l = self.__binary_search(x, e, tau)
                    x += l * ls_xe[e]
                    if sum(x) == self.k:
                        pbar.update(l)
                        self.memory = get_memory()
                        return x
                    pbar.update(int(l))
        self.memory = get_memory()
        return x
