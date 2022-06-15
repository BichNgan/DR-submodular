from datetime import datetime
import time

from loguru import logger

from algorithms import Algorithm2
from result import *
from configs import *
from tools import read_dataset
from objective_functions import budget_allocation, monotone_reduction

if __name__ == "__main__":
    init_config()
    logger.add(f"{log_dir}/alg2_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')}.log", 
               rotation="20 MB")
    for dataset in datasets:
        results = []
        logger.info(f'Processing {dataset}')
        E, pst = read_dataset(f'{data_dir}/{dataset}')
        n = len(E)
        f = budget_allocation(n, pst, max_workers)
        #f = monotone_reduction(n)
        for b, k in zip(b_values, k_values):
            f.reset()
            B = np.full(n, b)
            start = time.time()
            alg2 = Algorithm2(e_arr=E, b_arr=B, f=f, k=k, epsilon=epsilon)
            x = alg2.run()
            duration = time.time() - start
            results.append(to_result(epsilon, b, k, 
                                     f.count, alg2.memory, n,
                                     duration,f(x),
                                     np.sum(x),
                                     len(x[x>0])
                                     ))
        logger.info(results[0])
        df = to_pandas(results)
        save_result(df, f'{output_dir}/alg2_{dataset}.csv')
