import argparse
import inspect
import json
import numpy as np
import os
import pandas as pd
from scipy.special import softmax
import sys
from tqdm.auto import tqdm

from src.eval.Scorer import Scorer
from src.utils.Config import Config


notebook = sys.argv[0].endswith('ipykernel_launcher.py')
sm2name = {
    'mdl': 'MDL',
    'cv': 'CV',
    'Dev': 'Dev',
    'Mean': 'Mean',
    'Median': 'Median',
    'Best': 'Best',
    'Worst': 'Worst',
}
tn2stats = {
    'BoolQ': ['Acc'],
    'CB': ['Acc', 'F1'],
    'COPA': ['Acc'],
    'MultiRC': ['EM', 'F1'],
    'ReCoRD': ['EM', 'F1'],
    'RTE': ['Acc'],
    'WiC': ['Acc'],
    'WSC': ['Acc'],
    'Avg': [''],
}
tn2num_lines = {
    'BoolQ': 9427,
    'CB': 250,
    'COPA': 400,
    'MultiRC': 456,
    'ReCoRD': 65709,
    'RTE': 2490,
    'WiC': 5428,
    'WSC': 259,
}
tn2examples_per_train = {
    'MultiRC': 5100 / tn2num_lines['MultiRC'],
    'ReCoRD': 101000 / tn2num_lines['ReCoRD'],
}


def plot_results_by_num_train_adapet(tn2sm2str, tn, sms, stat, min_num_train, num_trains, examples_per_train, notebook=False, save_dir=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    xs = np.round(np.exp(np.linspace(np.log(min_num_train), np.log(tn2num_lines[tn]), num_trains)) * examples_per_train).astype(int)
    xs = xs[:min([len(tn2sm2str[tn][sm][stat]) for sm in sms])]
    plt.plot(xs, tn2sm2str[tn]['Mean'][stat], label=sm2name['Mean'], color='k')
    plt.fill_between(xs, tn2sm2str[tn]['Worst'][stat], tn2sm2str[tn]['Best'][stat], color='gray', alpha=0.3, linewidth=0)
    for sm_no, sm in enumerate(sms):
        stat2scores = tn2sm2str[tn][sm]
        plt.plot(xs, stat2scores[stat], label=sm2name[sm], color=plt.cm.plasma(sm_no / 2))
    plt.title(f'$\\bf{{{tn}}}$', fontsize=16)
    plt.xlim(xmin=min(xs), xmax=max(xs))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xscale('log')
    plt.ylabel(("Accuracy" if stat == "Acc" else stat) + ' (%)', fontsize=16)
    plt.xlabel('Train Size', fontsize=16)
    ax.set_xticks([], minor=True)
    ax.set_xticks(xs)
    ax.set_xticklabels(xs, fontsize=14, rotation=30, ha="right")
    plt.legend(fontsize=16)
    if save_dir is not None:
        save_file = f'{save_dir}/{inspect.stack()[0][3]}.tn-{tn}.stat-{stat}.pdf'
        plt.savefig(save_file, bbox_inches='tight')
        print('Saving to:', save_file)
    if notebook:
        plt.show()


def get_stats(save_dir, i, iter_no, tss, eps=1e-6, load_separate_dev_scores=True):
    with open(f'{save_dir}/iter-{iter_no}.dev_pred.txt') as f:
        results = json.load(f)

    out = {}
    for split in ['train'] + ([] if 'fewglue/ReCoRD' in save_dir else ['dev']):
        if 'fewglue/ReCoRD' in save_dir:
            labels_filebase = f'eval_train_labels.seed-{tss}' if split == 'train' else 'dev_labels'
            with open(f'data/fewglue/ReCoRD/{labels_filebase}.json') as f:
                idx2labels = json.load(f)
            
            acc_cor_cnt = 0
            acc_ttl_cnt = 0
            nlls = []
            for (idx, pred_true_lbl) in results[split].items():
                pred_lbl, true_lbl, logits = pred_true_lbl[0]
                assert len(pred_true_lbl) == len(idx2labels[idx]), f'Expected len(pred_true_lbl) ({len(pred_true_lbl)}) == len(idx2labels[idx]) ({len(idx2labels[idx])})'
                labels = idx2labels[idx][0]
                assert len(labels) == len(logits)
                assert pred_lbl in {0, 1}, f'Expected pred_lbl ({pred_lbl}) in {{0, 1}}'
                assert true_lbl == 1, f'Expected true_lbl ({true_lbl}) == 1'
                nlls.append(-np.log(eps if pred_lbl == 0 else (1. - eps)))
                acc_ttl_cnt += 1
                if pred_lbl == true_lbl:
                    acc_cor_cnt += 1
            out[f'computed_{split}_acc'] = acc_cor_cnt / acc_ttl_cnt
            out[f'computed_{split}_loss'] = np.sum(nlls)
        elif 'fewglue/MultiRC' in save_dir:
            if split not in results:
                continue
            acc_cor_cnt = 0
            acc_ttl_cnt = 0
            nlls = []
            for (idx, pred_true_lbl) in results[split].items():
                exact_match = True
                ex_nlls = []
                for (pred_lbl, true_lbl, probs) in pred_true_lbl:
                    if pred_lbl != true_lbl:
                        exact_match = False
                    ex_nlls.append(-np.log(probs[true_lbl] / sum(probs)))
                nlls.append(np.sum(ex_nlls))
                if exact_match:
                    acc_cor_cnt += 1
                acc_ttl_cnt += 1
            out[f'computed_{split}_acc'] = acc_cor_cnt / acc_ttl_cnt
            out[f'computed_{split}_loss'] = np.sum(nlls)
        else:
            if split not in results:
                continue
            split_results = np.array(list(results[split].values()))
            if 'fewglue/COPA' not in save_dir:
                assert split_results.shape[1] == 1, f'split_results.shape == {split_results.shape} | save_dir: {save_dir}'
            pred_lbl, true_lbl, probs = np.transpose(split_results[:, 0, :], (1, 0))
            if isinstance(true_lbl[0], bool):
                true_lbl = true_lbl.astype(int)
            probs = np.array([p for p in probs]).astype(float)
            if 'fewglue/COPA' in save_dir:  # probs are actually logits
                assert len(probs.shape) == 2, f'Expected len(probs.shape) ({len(probs.shape)}) == 2'
                probs = softmax(probs, axis=-1)
            # Smooth probabilities
            probs[probs == 0] = eps
            probs[probs == 1] = 1. - eps

            assert np.all(probs > 0), f'Expected positive probs but got: {probs}'
            assert np.all(probs < 1), f'Expected sub-1 probs but got: {probs}'
            nlls = -np.log(probs / np.expand_dims(probs.sum(1), axis=1))

            num_mismatched_preds = (probs.argmax(axis=1) != pred_lbl).sum()
            assert num_mismatched_preds == 0, f'Expected num_mismatched_preds ({num_mismatched_preds}) == 0'

            out[f'computed_{split}_acc'] = np.mean(pred_lbl == true_lbl)
            out[f'computed_{split}_loss'] = np.sum([nll[true] for nll, true in zip(nlls, true_lbl)])
        
    # Add ADAPET officially computed scores and losses
    if ('fewglue/ReCoRD' in save_dir) and load_separate_dev_scores:
        config = Config(os.path.join(save_dir, "config.json"), mkdir=False, update_exp_config=False)
        dev_pred_dir, dev_pred_file = config.dev_pred_file.rsplit('/', 1)

        scorer = Scorer(config, "fewglue/ReCoRD")
        with open(f'{save_dir}/eval.iter-{iter_no}.{dev_pred_file}') as f:
            scorer.dict_idx2logits_lbl = json.load(f)['dev']
        for k, v in scorer.get_score("dev")[1].items():
            out[k] = v
    else:
        with open(f'{save_dir}/dev_scores.json') as f:
            for k, v in [json.loads(line) for line in f][i].items():
                out[k] = v
        
    return out


parser = argparse.ArgumentParser()
parser.add_argument('--exp', default='super_glue', type=str, choices=['super_glue', 'vary_num_train'])
parser.add_argument('--tns', default=('BoolQ', 'CB', 'COPA', 'RTE', 'WiC', 'WSC', 'MultiRC', 'ReCoRD'), nargs='+', type=str, help='Task names')
parser.add_argument('--tsss', default=(0, 1, 2, 3), nargs='+', type=int, help='Train set seeds')
parser.add_argument('--sms', default=('cv', 'mdl'), nargs='+', type=str, help='Selection methods (criteria) to test')
parser.add_argument('--min_num_train', default=32, type=int)
parser.add_argument('--num_trains', default=8, type=int)
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args('') if notebook else parser.parse_args()
"""
The default arguments above reproduce our main ADAPET results table.

For experiments varying the number of training examples, we used:
    "python adapet.py --exp vary_num_train --tns 'MultiRC' 'WiC' 'BoolQ' --tsss 1 11 21 31 41 51 61 71 --sms cv"
    where different train set seeds passed into "--tsss" use increasing amounts of data
"""



exp = args.exp
min_num_train = args.min_num_train
num_trains = args.num_trains
tns = args.tns
tsss = args.tsss
sms = args.sms
fmrs = [False]
mas = ['0.075', '0.10', '0.105', '0.15']
nb = 8
iters = [0, 1, 2, 3]  # early stopping checkpoint number
tn2gaf = {tn: json.load(open(f'{os.getenv("BASE")}/config/{tn}.json'))['grad_accumulation_factor'] for tn in tns}


tn2sm2tss2fmr2ma2bn2iter = {}
for tn in tqdm(tns):
    tn2sm2tss2fmr2ma2bn2iter[tn] = {}
    for sm in sms:
        tn2sm2tss2fmr2ma2bn2iter[tn][sm] = {}
        for tss in tsss:
            tn2sm2tss2fmr2ma2bn2iter[tn][sm][tss] = {}
            for fmr in fmrs:
                tn2sm2tss2fmr2ma2bn2iter[tn][sm][tss][fmr] = {}
                for ma in mas:
                    tn2sm2tss2fmr2ma2bn2iter[tn][sm][tss][fmr][ma] = {}
                    for bn in range(nb):
                        tn2sm2tss2fmr2ma2bn2iter[tn][sm][tss][fmr][ma][bn] = {}
                        save_dir = f'{os.getenv("BASE")}/exp_out/fewglue/{tn}/albert-xxlarge-v2/tss-{tss}.ma-{ma}.fmr-{fmr}.sm-{sm}.nb-{nb}.bn-{bn}'
                        for i in iters:
                            iter_no = (((i + 1) * 250) - 1) * tn2gaf[tn]
                            tn2sm2tss2fmr2ma2bn2iter[tn][sm][tss][fmr][ma][bn][i] = get_stats(save_dir, i, iter_no, tss, load_separate_dev_scores=False)


tn2tss2fmr2ma2iter = {}
for tn in tns:
    tn2tss2fmr2ma2iter[tn] = {}
    for tss in tsss:
        tn2tss2fmr2ma2iter[tn][tss] = {}
        for fmr in fmrs:
            tn2tss2fmr2ma2iter[tn][tss][fmr] = {}
            for ma in mas:
                tn2tss2fmr2ma2iter[tn][tss][fmr][ma] = {}
                save_dir = f'{os.getenv("BASE")}/exp_out/fewglue/{tn}/albert-xxlarge-v2/tss-{tss}.ma-{ma}.fmr-{fmr}'
                for i in iters:
                    iter_no = (((i + 1) * 250) - 1) * tn2gaf[tn]
                    tn2tss2fmr2ma2iter[tn][tss][fmr][ma][i] = get_stats(save_dir, i, iter_no, tss)


selection_stat = 'computed_train_loss'
tn2sm2str = {}
tn2sm2stat2scores = {tn: {} for tn in tns}
for tn in tns:
    if args.verbose:
        print('*' * 20, '\n', tn)
    tn2sm2str[tn] = {}
    for sm in sms:
        stat2scores = {stat: [] for stat in tn2stats[tn]}
        for tss in tsss:
            best_hps = []
            best_score = float('inf')
            for fmr in fmrs:  # True
                for ma in mas:
                    for i in iters:
                        score = np.sum([tn2sm2tss2fmr2ma2bn2iter[tn][sm][tss][fmr][ma][bn][i][selection_stat] for bn in range(nb)])
                        if exp == 'super_glue':
                            score /= min_num_train
                        if score < best_score:
                            best_score = score
                            best_hps = [(fmr, ma, i)]
                        elif score == best_score:
                            best_hps.append((fmr, ma, i))
            assert len(best_hps) > 0, f'Expected len(best_hps) ({len(best_hps)}) > 0'
            best_hp_scores = [tn2tss2fmr2ma2iter[tn][tss][fmr][ma][i] for fmr, ma, i in best_hps]
            for stat in tn2stats[tn]:
                stat2scores[stat].append(100. * np.mean([best_hp_score['dev_acc' if (tn in {'MultiRC', 'ReCoRD'} and stat.lower()) == 'em' else f'dev_{stat.lower()}'] for best_hp_score in best_hp_scores]))
            if args.verbose:
                print('\t\t', best_hps, '\t|', round(best_score, 5))
        tn2sm2stat2scores[tn][sm] = stat2scores
        tn2sm2str[tn][sm2name[sm]] = ''
        for stat in tn2stats[tn]:
            if len(tn2sm2str[tn][sm2name[sm]]) > 0:
                tn2sm2str[tn][sm2name[sm]] += '/'
            if args.verbose:
                print(stat, '\t', sm2name[sm], '\t', round(np.mean(stat2scores[stat]), 1), '+/-', round(np.std(stat2scores[stat], ddof=1), 1), best_hps)
            tn2sm2str[tn][sm2name[sm]] += str(round(np.mean(stat2scores[stat]), 1))
            if len(tsss) > 1:
                tn2sm2str[tn][sm2name[sm]] += f'$_{{{round(np.std(stat2scores[stat], ddof=1), 1)}}}$'


for sm in ['Best', 'Worst', 'Mean', 'Median']:
    if args.verbose:
        print(sm)
    for tn in tns:
        if args.verbose:
            print(tn)
        stat2scores = {stat: [] for stat in tn2stats[tn]}
        scores = []
        for tss in tsss:
            best_hps = []
            best_score = float('-inf') if sm == 'Best' else float('inf')
            for fmr in fmrs:
                for ma in mas:
                    for i in iters:
                        score = tn2tss2fmr2ma2iter[tn][tss][fmr][ma][i][f'dev_{tn2stats[tn][0].replace("EM", "Acc").lower()}']
                        if ((sm == 'Best') and (score > best_score)) or ((sm == 'Worst') and (score < best_score)):
                            best_score = score
                            best_hps = [(fmr, ma, i)]
                        elif (sm in {'Mean', 'Median'}) or (score == best_score):
                            best_hps.append((fmr, ma, i))
            assert len(best_hps) > 0, f'Expected len(best_hps) ({len(best_hps)}) > 0'
            best_hp_scores = [tn2tss2fmr2ma2iter[tn][tss][fmr][ma][i] for fmr, ma, i in best_hps]
            if sm in {'Mean', 'Median'}:
                assert len(best_hp_scores) == len(fmrs) * len(mas) * len(iters)
            for stat in tn2stats[tn]:
                stat_scores = 100. * np.array([best_hp_score[f'dev_{stat.replace("EM", "Acc").lower()}'] for best_hp_score in best_hp_scores])
                stat2scores[stat].append(np.median(stat_scores) if sm == 'Median' else np.mean(stat_scores))
            if (sm not in {'Mean', 'Median'}) and args.verbose:
                print('\t\t', best_hps, '\t|', round(best_score, 5))
        tn2sm2stat2scores[tn][sm] = stat2scores
        tn2sm2str[tn][sm2name[sm]] = ''
        for stat in tn2stats[tn]:
            if len(tn2sm2str[tn][sm2name[sm]]) > 0:
                tn2sm2str[tn][sm2name[sm]] += '/'
            if args.verbose:
                print(stat, '\t', sm2name[sm], '\t', round(np.mean(stat2scores[stat]), 1), '+/-', round(np.std(stat2scores[stat], ddof=1), 1), best_hps)
            tn2sm2str[tn][sm2name[sm]] += str(round(np.mean(stat2scores[stat]), 1))
            if len(tsss) > 1:
                tn2sm2str[tn][sm2name[sm]] += f'$_{{{round(np.std(stat2scores[stat], ddof=1), 1)}}}$'


# Compute average SuperGLUE score
sm2all_tn_scores = {}
for tn, sm2stat2scores in tn2sm2stat2scores.items():
    for sm, stat2scores in sm2stat2scores.items():
        if sm not in sm2all_tn_scores:
            sm2all_tn_scores[sm] = []
        tn_scores = np.array(list(stat2scores.values())).mean(0)
        assert len(tn_scores.shape) == 1, f'Expected len(tn_scores.shape) ({len(tn_scores.shape)}) == 1'
        sm2all_tn_scores[sm].append(tn_scores)

tn2sm2str['Avg'] = {}
for sm, all_tn_scores in sm2all_tn_scores.items():
    avg_scores = np.array(all_tn_scores).mean(0)
    assert len(avg_scores.shape) == 1, f'Expected len(avg_scores.shape) ({len(avg_scores.shape)}) == 1'
    tn2sm2str['Avg'][sm2name[sm]] = f'{round(np.mean(avg_scores), 1)}$_{{{round(np.std(avg_scores, ddof=1), 1)}}}$'


if exp == 'super_glue':
    table = pd.DataFrame({(f'\textbf{{{tn}}}', "/".join(tn2stats[tn])): sm2str for tn, sm2str in tn2sm2str.items()}).sort_index(ascending=False)
    if notebook:
        from IPython.display import display
        display(table)
    print(table.to_latex(escape=False, bold_rows=True, column_format='l' + 'c' * len(tn2sm2str)))
elif exp == 'vary_num_train':
    save_dir = '../prompt_rda/plots/adapet'
    os.makedirs(save_dir, exist_ok=True)
    for tn in tns:
        for stat in tn2stats[tn]:
            plot_results_by_num_train_adapet(tn2sm2stat2scores, tn, sms, stat, min_num_train, num_trains, tn2examples_per_train.get(tn, 1), notebook, save_dir)
