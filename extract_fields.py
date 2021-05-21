import argparse
from tqdm.auto import tqdm
import json
from multiprocessing import Pool, cpu_count
import numpy as np
import os
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--num_trains", default=[5, 10, 15, 20, 30, 40], nargs='+', type=int, help="The number of training examples to use.")
parser.add_argument("--data_name", default='TREx', choices=['TREx', 'Google_RE', 'super_glue'], type=str, help="The dataset to use.")
parser.add_argument("--engines", default=['distilgpt2', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], nargs='+', type=str, help="The engine (model) to use.")
parser.add_argument("--seeds", default=[0, 1, 2, 3, 4], nargs='+', type=int, help="The random seeds to use for selecting training data points.")
parser.add_argument("--num_dev", default=None, type=int, help="The number of dev examples to use (set automatically if not provided).")
parser.add_argument("--keys", default=['nlls', 'ranks'], nargs='+', type=str, help="The keys of data to extract and save.")
args = parser.parse_args('') if sys.argv[0].endswith('ipykernel_launcher.py') else parser.parse_args()

keys = tuple(args.keys)


def save_keys(filebase):
    """Save template2results in a file that can be load with dict(np.load(save_file))"""
    with open(f'{filebase}.json') as f:
        template2results = json.load(f)
    for k in keys:
        np.savez(f'{filebase}.key-{k}.npz', **{template: results[k] for template, results in template2results.items()})


filebases = []
for num_train in args.num_trains:
    for engine in args.engines:
        assert (num_train > 0) or (args.num_dev is None), 'Not implemented using args.num_dev != None for few-shot evaluation'
        save_dir = f'{os.getenv("BASE")}/data/rel2template2results.data_name-{args.data_name}_UHN.engine-{engine}.num_train-{num_train}.sort_by_weight-False'
        if args.num_dev is not None:
            save_dir += f'.num_dev-{args.num_dev}'

        for seed in args.seeds:
            rel_filebases = []
            for rel_file in os.listdir(f'{save_dir}/seed-{seed}'):
                if rel_file.endswith('.json') and ((args.data_name != 'TREx') or (rel_file[0] == 'P')):
                    rel_filebases.append(f'{save_dir}/seed-{seed}/{rel_file[:-len(".json")]}')
            if args.data_name == 'TREx':
                assert len(rel_filebases) == 41, f'Expected len(rel_filebases) ({len(rel_filebases)}) == 41 relations'
            for filebase in rel_filebases:
                if not all([os.path.exists(f'{filebase}.key-{k}.npz') for k in keys]):
                    filebases.append(filebase)


processes = int(os.environ.get('SLURM_CPUS_PER_TASK', cpu_count()))
print('# Processes:', processes)
with Pool(processes=processes) as p:
    with tqdm(total=len(filebases)) as pbar:
        for i, _ in enumerate(p.imap_unordered(save_keys, filebases)):
            pbar.update()
