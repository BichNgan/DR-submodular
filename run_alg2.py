from datetime import datetime
import time
import json

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
    for dataset, delimiter in zip(datasets, delimiters):
        results = []
        logger.info(f'Processing {dataset}')
        E, pst, is_source = read_dataset(f'{data_dir}/{dataset}', delimiter=delimiter)
        n = len(E)
        #f = monotone_reduction(n)
        #E = list(range(n))
        f = budget_allocation(n, pst)
        for b, k in zip(b_values, k_values):
            logger.info(f'### Running for b={b} and k={k} ###')
            f.reset() 
            B = np.full(n, b)
            start = time.time()
            alg2 = Algorithm2(e_arr=E, b_arr=B, f=f, k=k, 
                              epsilon=epsilon, is_source=is_source)
            x = alg2.run()
            duration = time.time() - start
            results.append(to_result(epsilon, int(b), int(k), 
                                     f.count, alg2.memory, n,
                                     duration, float(f(x)),
                                     int(np.sum(x)),
                                     len(x[x>0])
                                     ))
            json_data = {
                    'alg': 'alg2',
                    'data': results[-1]
                    }
            logger.info(f"""
            -------Result-------
            {json.dumps(json_data)}
            --------------------""")
            df = to_pandas(results)
            save_result(df, f'{output_dir}/alg2_{dataset}.csv')
