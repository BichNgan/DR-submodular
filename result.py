import pandas as pd

class ResultDictSchema:
    def __init__(self):
        self.__schema = ['bound', 'n_oracles', 'memory', 
                         'n', 'time', 'influence']
    def get_schema(self):
        return self.__schema

    def to_result(self, bound, n_oracles, memory, n, time, influence):
        return {'bound': bound,
                'n_oracles': n_oracles,
                'memory': memory,
                'n': n,
                'time': time,
                'influence': influence}
        
def to_pandas(results):
    schema = ResultDictSchema().get_schema()
    tmp = dict.fromkeys(schema, [])
    for result in results:
        for key in result:
            tmp[key].append(result[key])
    return pd.DataFrame(tmp)

def save_result(df, file_name):
    df.to_csv(file_name, index=False)      

