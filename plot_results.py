import argparse
from IPython.display import display, HTML
import math
import numpy as np
import os
import sys

from utils import generate_permutations
from plot_utils import load_results, compute_generalization_estimates, compute_stats, plot_compute_efficiency, plot_prompt_transfer, plot_results_by_engine, plot_results_by_num_train, plot_results_distribution, plot_results_distribution_by_stat


notebook = sys.argv[0].endswith('ipykernel_launcher.py')
display(HTML("<style>.container { width:100% !important; }</style>"))


parser = argparse.ArgumentParser()
parser.add_argument('--exp', default='TREx-vary_models')
parser.add_argument('--use_gpt3', action='store_true')
args = parser.parse_args('') if notebook else parser.parse_args()

exp = args.exp
gpt2_engines = ['distilgpt2', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
gpt3_engines = ['ada', 'babbage', 'curie', 'davinci'] if args.use_gpt3 else []
exp2config = {
    'TREx-vary_models': {
        'data_name': 'TREx',
        'engines': gpt2_engines + gpt3_engines,
        'num_trains': [5],
    },
    'TREx-vary_num_train': {
        'data_name': 'TREx',
        'engines': gpt2_engines + gpt3_engines[:2],
        'num_trains': [5, 10, 15, 20, 30, 40],
    },
    'TREx-vary_criterion': {
        'data_name': 'TREx',
        'engines': gpt2_engines + gpt3_engines,
        'num_trains': [5],
        'plot_stat_names': ['example_nlls_$\\beta=0.0$', 'example_nlls_$\\beta=100000000.0$', 'nlls_$\\beta=0.0$', 'nlls_$\\beta=1.0$', 'nlls_$\\beta=100000000.0$', 'bayes'],
    },
    'RTE': {
        'data_name': 'RTE',
        'engines': gpt2_engines + gpt3_engines,
        'num_trains': [5],
    },
    'CB': {
        'data_name': 'CB',
        'engines': gpt2_engines + gpt3_engines,
        'num_trains': [5],
    },
    'WiC': {
        'data_name': 'WiC',
        'engines': gpt2_engines + gpt3_engines,
        'num_trains': [5],
    },
}
num_trains = exp2config[exp]['num_trains']
engines = exp2config[exp]['engines']
data_name = exp2config[exp]['data_name']
plot_stat_names = exp2config[exp].get('plot_stat_names', ['nlls_$\\beta=0.0$', 'nlls_$\\beta=100000000.0$'])


num_dev = None
seeds = [0, 1, 2, 3, 4]
rels = [data_name] if data_name != 'TREx' else ['P937', 'P740', 'P530', 'P527', 'P495', 'P47', 'P463', 'P449', 'P413', 'P407', 'P39', 'P37', 'P364', 'P361', 'P36', 'P31', 'P30', 'P279', 'P276', 'P27', 'P264', 'P20', 'P190', 'P19', 'P178', 'P176', 'P17', 'P159', 'P1412', 'P140', 'P138', 'P1376', 'P136', 'P131', 'P1303', 'P127', 'P108', 'P106', 'P103', 'P101', 'P1001']
keys = ['nlls', 'ranks'] if data_name == 'TREx' else ['verbalizer_nlls', 'verbalizer_accs']
if any(['example_nlls' in s for s in plot_stat_names]):
    keys.append('example_nlls')
betas = [0.0, 1e8]  # 0 for MDL, 1e8 for LOOCV
if any([('$\\beta=1.0$' in s) or ('$\\beta=1$' in s) for s in plot_stat_names]):
    betas.insert(1, 1.0)
assert not (('nlls' in keys) and ('verbalizer_nlls' in keys))
assert (max(num_trains) > 0) or (num_dev is not None), f'Not implemented: Using few-shot evaluation with num_dev != None'
plot_data_name = 'LAMA-UHN' if data_name == 'TREx' else data_name
max_num_samples = math.factorial(min(num_trains))
num_train2num_samples = {num_train: np.array([f for f in range(1, max_num_samples + 1) if ((max_num_samples % f) == 0) and ((f % num_train == 0) or (f == 1))]).astype(int) for num_train in num_trains}
load_data_name = 'super_glue' if data_name != 'TREx' else data_name
num_total_dev = 56 if exp == 'CB' else None
save_dir = f'{os.getenv("BASE")}/plots/{exp}'
os.makedirs(save_dir, exist_ok=True)
plot_data_name = 'LAMA-UHN' if data_name == 'TREx' else data_name


num_train2engine2seed2rel2template2results = load_results(load_data_name, engines, num_trains, num_dev, seeds, rels, keys)
num_train2engine2seed2rel2stat2results = compute_generalization_estimates(num_train2engine2seed2rel2template2results, load_data_name, engines, num_trains, num_dev, rels, keys, betas)
num_train2cyclic_permutations = {num_train: generate_permutations(num_train, num_dev, load_data_name)[1] for num_train in num_trains}


plot_type2num_train2engine2engine2stat2results = compute_stats(num_train2engine2seed2rel2stat2results, num_train2num_samples, num_train2cyclic_permutations, data_name, engines, num_trains, rels, plot_stat_names + ['test_acc'], ['Test Accuracy of Chosen Prompt (%)'], num_total_dev, test_transfer=True)


# Plot Compute Efficiency
plot_type = 'Test Accuracy of Chosen Prompt (%)'
scale = 'rel'
plot_num_estimates = False

for num_train in num_trains:
    for engine in [engines[-1]]:  # Show efficiency for largest model only (for brevity)
        plot_compute_efficiency(plot_type2num_train2engine2engine2stat2results, plot_type, num_train, engine, engine, plot_stat_names, num_train2num_samples, scale, plot_num_estimates, plot_data_name, show_legend=(num_train == 5), show_y=(num_train == 5), notebook=notebook, save_dir=save_dir if (engine == engines[-1]) and (len(num_trains) > 1) else None)


# Test prompt transfer
plot_type = 'Test Accuracy of Chosen Prompt (%)'
scale = 'rel'
num_samples = 'Multi'

for stat in plot_stat_names + ['test_acc']:
    for num_train in num_trains:
        plot_prompt_transfer(plot_type2num_train2engine2engine2stat2results, plot_type, num_train, engines, stat, num_train2num_samples, scale, num_samples, plot_data_name, show_y=(stat == plot_stat_names[0]), show_cbar=(stat == 'test_acc'), save_dir=save_dir if exp == 'TREx-vary_models' else None)


plot_types = ['Test Accuracy of Chosen Prompt (%)', 'Accuracy at Choosing Best Prompt (%)', 'Frequency of Choosing Worst Prompt (%)']
acc_plot_type2num_train2engine2engine2stat2results = compute_stats(num_train2engine2seed2rel2stat2results, num_train2num_samples, num_train2cyclic_permutations, data_name, engines, num_trains, rels, plot_stat_names, plot_types, num_total_dev, test_transfer=False)


scale = 'rel'

for plot_type in ['Test Accuracy of Chosen Prompt (%)', 'Accuracy at Choosing Best Prompt (%)', 'Frequency of Choosing Worst Prompt (%)']:
    if plot_type == 'Accuracy at Choosing Best Prompt (%)':
        if scale == 'rel':
            continue
        mean_acc = 100. * np.mean([1. / len(template2results) for template2results in num_train2engine2seed2rel2template2results[num_trains[0]][engines[0]][seeds[0]].values()]) # Mean accuracy
        top = {
            'TREx': 37 if exp == 'TREx-vary_criterion' else 35,
            'RTE': 100,
        }.get(data_name, 100)
        bottom = {
            'TREx': mean_acc,
            'RTE': 0,
        }.get(data_name, 0)

        print('Mean Accuracy (%)', mean_acc)
    elif plot_type == 'Frequency of Choosing Worst Prompt (%)':
        if scale == 'rel':
            continue
        mean_acc = 100. * np.mean([1. / len(template2results) for template2results in num_train2engine2seed2rel2template2results[num_trains[0]][engines[0]][seeds[0]].values()]) # Mean accuracy
        top = {
            'TREx': 35,
        }.get(data_name, 100)
        bottom = {
        }.get(data_name, 0)

        print('Mean Accuracy (%)', mean_acc)
    else:
        top = {
            'TREx': 62,
            'RTE': 70,
            'CB': 80,
            'WiC': 54.5,
            'BoolQ': 80,
        }.get(data_name, 100)
        bottom = {
            'TREx': 0,
            'RTE': 50.3 - 5,
            'CB': 48.4 - 10,
            'WiC': 50.0 - 1.5,
            'BoolQ': 62.3 - 20,
        }.get(data_name, 0)
    for num_samples in ['Multi']:
        for num_train in num_trains:
            plot_results_by_engine(acc_plot_type2num_train2engine2engine2stat2results, plot_type, num_train, engines, plot_stat_names, num_train2num_samples, scale, num_samples, top, bottom, plot_data_name, show_legend=exp in {'TREx-vary_models', 'TREx-vary_criterion'}, show_y=exp in {'TREx-vary_models', 'TREx-vary_criterion', 'RTE'}, show_legend_title=plot_data_name != 'LAMA-UHN', legend_bbox_to_anchor=((1, 1) if exp=='TREx-vary_criterion' else None), figsize=((16, 4) if exp=='TREx-vary_criterion' else None), notebook=notebook, save_dir=save_dir if exp in {'TREx-vary_models', 'TREx-vary_criterion', 'RTE', 'CB', 'WiC', 'BoolQ'} else None)


scale = 'rel'

for plot_type in ['Test Accuracy of Chosen Prompt (%)', 'Accuracy at Choosing Best Prompt (%)']:
    if plot_type == 'Accuracy at Choosing Best Prompt (%)':
        if scale == 'rel':
            continue
        mean_acc = 100. * np.mean([1. / len(template2results) for template2results in num_train2engine2seed2rel2template2results[num_trains[0]][engines[0]][seeds[0]].values()]) # Mean accuracy
        top = {
            'TREx': 35,
            'RTE': 100,
        }.get(data_name, 100)
        bottom = {
            'TREx': mean_acc,
            'RTE': 0,
        }.get(data_name, 0)

        print('Mean Accuracy (%)', mean_acc)
    else:
        top = {
            'TREx': 100 if scale == 'rel' else 62,
            'RTE': 70,
            'CB': 80,
            'WiC': 54.5,
            'BoolQ': 80,
        }.get(data_name, 100)
        bottom = {
            'TREx': -20 if scale == 'rel' else 0,
            'RTE': 50.3 - 5,
            'CB': 48.4 - 10,
            'WiC': 50.0 - 1.5,
            'BoolQ': 62.3 - 20,
        }.get(data_name, 0)
    for num_samples in ['Multi']:
        for num_train in num_trains:
            plot_results_by_engine(acc_plot_type2num_train2engine2engine2stat2results, plot_type, num_train, engines, plot_stat_names, num_train2num_samples, scale, num_samples, top, bottom, plot_data_name, show_legend=exp in {'TREx-vary_models'}, show_y=exp in {'TREx-vary_models', 'RTE'}, show_legend_title=not (plot_data_name == 'LAMA-UHN' and scale == 'rel'), notebook=notebook, save_dir=save_dir if exp in {'TREx-vary_models', 'RTE', 'CB', 'WiC', 'BoolQ'} else None)


if len(num_trains) > 1:
    for plot_type in plot_types:
        scale = 'rel' if plot_type == 'Test Accuracy of Chosen Prompt (%)' else 'abs'
        for num_samples in ['Multi']:
            for stat in plot_stat_names:
                plot_results_by_num_train(acc_plot_type2num_train2engine2engine2stat2results, plot_type, num_trains, engines, stat, num_train2num_samples, scale, num_samples, show_legend=(plot_type == plot_types[-1]), show_y=True, notebook=notebook, save_dir=save_dir if exp == 'TREx-vary_num_train' else None)


plot_type = 'Test Accuracy of Chosen Prompt (%)'
scale = 'diff'
plot_info = 'cdf'

for num_samples in ['Multi'] if data_name == 'TREx' else ['One']:
    for num_train in num_trains:
        gains = plot_results_distribution(acc_plot_type2num_train2engine2engine2stat2results, plot_type, num_train, engines, plot_stat_names, num_train2num_samples, scale, num_samples, plot_info, plot_data_name, show_legend=plot_data_name in {'LAMA-UHN', 'RTE'}, show_y=plot_data_name in {'LAMA-UHN', 'RTE'}, notebook=notebook, save_dir=save_dir if exp in {'TREx-vary_models', 'RTE', 'CB', 'WiC', 'BoolQ'} else None)


plot_type = 'Test Accuracy of Chosen Prompt (%)'
scale = 'diff'
plot_info = 'cdf'

for num_samples in ['Multi'] if data_name == 'TREx' else ['One']:
    for num_train in num_trains:
        for stat in plot_stat_names[-1:]:
            gains = plot_results_distribution_by_stat(acc_plot_type2num_train2engine2engine2stat2results, plot_type, num_train, engines, [f'{stat}{postfix}' for postfix in ["_$\\alpha=3$", "_$\\alpha=2$", "_$\\alpha=1$", '']], num_train2num_samples, scale, num_samples, plot_info, plot_data_name, show_y=False, notebook=notebook, save_dir=save_dir if exp == 'TREx-vary_models' else None)
