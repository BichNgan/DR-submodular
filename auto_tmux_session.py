import os 
import sys
from configs import Configuration

dataset = 'wiki'

if len(sys.argv) > 1:
    dataset = str(sys.argv[1])

config = Configuration()

tmux_session = (
        lambda alg, ds, k: 
        f"tmux new-session -d -s {alg}_{ds}_k{k} 'python3 run_params.py {alg} {ds} {k}'"
        )

for alg in config.get_alg_ids():
    for k in config.k_values:
        os.system(tmux_session(alg, dataset, k))

