import torch
import datetime
import os
import sys
import argparse
import subprocess
from shutil import copytree, ignore_patterns
import random
import numpy as np
import logging
import re
import sys
import json

global device; device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_dir(dir_name):
    '''
    Makes a directory if it doesn't exists yet
    Args:
        dir_name: directory name
    '''
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def make_exp_dir(base_exp_dir):
    '''
    Makes an experiment directory with timestamp
    Args:
        base_output_dir_name: base output directory name
    Returns:
        exp_dir_name: experiment directory name
    '''
    now = datetime.datetime.now()
    ts = "{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}".format(now.year, now.month, now.day, now.hour, now.minute,
                                                            now.second)
    exp_dir_name = os.path.join(base_exp_dir, ts)
    make_dir(exp_dir_name)

    src_file = os.path.join(exp_dir_name, 'src')

    # copytree(os.path.join(os.environ['PET_ELECTRA_ROOT'], "src"), src_file,  ignore=ignore_patterns('*.pyc', 'tmp*'))

    return exp_dir_name

def print_mem_usage(loc):
    '''
    Print memory usage in GB
    :return:
    '''
    print("%s mem usage: %.3f GB, %.3f GB, %.3f GB" % (loc, float(torch.cuda.memory_allocated() / 1e9), float(torch.cuda.memory_reserved() / 1e9),  float(torch.cuda.max_memory_allocated() / 1e9)))
    sys.stdout.flush()

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value


def update_dict_val_store(dict_val_store, dict_update_val, grad_accumulation_factor):
    '''
    Update dict_val_store with dict_update_val

    :param dict_val_store:
    :param dict_update_val:
    :return:
    '''
    if dict_val_store is None:
        dict_val_store = dict_update_val
    else:
        for k in dict_val_store.keys():
            dict_val_store[k] += dict_update_val[k] / grad_accumulation_factor

    return dict_val_store

def get_avg_dict_val_store(dict_val_store, num_batches=100):
    '''
    Get average dictionary val

    :param dict_val_store:
    :param eval_every:
    :return:
    '''
    dict_avg_val = {}

    for k in dict_val_store.keys():
        dict_avg_val[k] = float(dict_val_store[k].detach().cpu().item() / num_batches)

    return dict_avg_val


def remove_extra_padding(list_input_ids):
    input_ids = torch.tensor(list_input_ids)
    assert input_ids.dim() == 2, f'Expected input_ids.dim() ({input_ids.dim()}) == 2. input_ids.size() = {input_ids.size()}. input_ids = {input_ids}'
    nonzero_idxs = input_ids.nonzero()
    if len(nonzero_idxs) > 0:
        orig_seq_len = input_ids.size(1)
        max_seq_len = nonzero_idxs[:, 1].max().item() + 1
        if max_seq_len < orig_seq_len:
            input_ids = input_ids[:, :max_seq_len]
    return input_ids


def get_subset(data, config, split, is_eval):
    assert isinstance(data, list), f'Expected type(data) ({type(data)}) == list'
    assert config.selection_method in ['mdl', 'cv', None], f"Expected config.selection_method ({config.selection_method}) in ['mdl', 'cv', None]"

    if (split != 'train') or (config.selection_method is None):
        return data

    if config.selection_method == 'cv':
        block_start_idxs = np.linspace(0, len(data), config.num_blocks + 1).astype(int)
        if is_eval:
            return data[block_start_idxs[config.block_no]: block_start_idxs[config.block_no + 1]]
        else:
            return data[:block_start_idxs[config.block_no]] + data[block_start_idxs[config.block_no + 1]:]
    elif config.selection_method == 'mdl':
        block_start_idxs = np.linspace(config.min_block_size, len(data), config.num_blocks + 1).astype(int)
        if is_eval:
            return data[block_start_idxs[config.block_no]: block_start_idxs[config.block_no + 1]]
        else:
            return data[:block_start_idxs[config.block_no]]


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)
