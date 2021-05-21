import inspect
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.special import softmax
from scipy.stats import kendalltau, sem
from tqdm.auto import tqdm
from utils import generate_permutations, statsign, heatmap, annotate_heatmap, engine2name


cm = plt.cm.plasma
stat2name = {
    'verbalizer_nlls': 'argmin -log $p_{clf}(y|x, t)$',
    'verbalizer_entropys': 'argmin $H_{clf}(y|x)$',
    'nlls': 'argmin -log $p(y|x, t)$',
    'nony_nlls': 'argmin -log $p(x, t)$',
    '-nony_nlls': 'argmax -log $p(x, t)$',
    'entropys': 'argmin $H(y|x)$',
    '-entropys': 'argmax $H(y|x)$',
    'example_nlls': 'argmin -log $p(x, t, y)$',
    'waic': 'WAIC',
    'waic_gibbs': 'WAIC (Gibbs)',
    'test_acc': 'Test Accuracy',
}
plot_stat2name = {
    'mean': 'Mean',
    'best': 'Best',
    'worst': 'Worst',
    'test_acc': 'Test Accuracy',
    'nlls_$\\beta=0.0$': 'MDL',
    'nlls_$\\beta=1.0$': 'MDL$_{\\beta=1}$',
    'nlls_$\\beta=100000000.0$': 'CV',
    'example_nlls_$\\beta=0.0$': 'MDL$_{x,y}$',
    'example_nlls_$\\beta=1.0$': 'MDL$_{x,y,\\beta=1}$',
    'example_nlls_$\\beta=100000000.0$': 'CV$_{x,y}$',
    'variances': 'NLL Variance',
    'waic_bayes': 'WAIC (Bayes)',
    'bayes': 'CV$_{Bayes}$',
    'waic_gibbs': 'WAIC (Gibbs)',
    'gibbs': 'Gibbs Loss',
    'nlls_$\\beta=0.0$_$\\alpha=1$': 'MDL+$1\sigma$',
    'nlls_$\\beta=1.0$_$\\alpha=1$': '$\\beta=1.0$+$1\sigma$',
    'nlls_$\\beta=100000000.0$_$\\alpha=1$': 'CV$_{\\alpha=1}$',
    'variances_$\\alpha=1$': 'NLL Variance+$1\sigma$',
    'waic_bayes_$\\alpha=1$': 'WAIC$+1\sigma$ (Bayes)',
    'bayes_$\\alpha=1$': 'Bayes Loss+$1\sigma$',
    'waic_gibbs_$\\alpha=1$': 'WAIC$+1\sigma$ (Gibbs)',
    'gibbs_$\\alpha=1$': 'Gibbs Loss+$1\sigma$',
    'nlls_$\\beta=0.0$_$\\alpha=1.5$': 'MDL+$1.5\sigma$',
    'nlls_$\\beta=1.0$_$\\alpha=1.5$': '$\\beta=1.0$+$1.5\sigma$',
    'nlls_$\\beta=100000000.0$_$\\alpha=1.5$': 'CV$_{\\alpha=1.5}$',
    'nlls_$\\beta=0.0$_$\\alpha=2$': 'MDL+$2\sigma$',
    'nlls_$\\beta=1.0$_$\\alpha=2$': '$\\beta=1.0$+$2\sigma$',
    'nlls_$\\beta=100000000.0$_$\\alpha=2$': 'CV$_{\\alpha=2}$',
    'variances_$\\alpha=2$': 'NLL Variance+$2\sigma$',
    'waic_bayes_$\\alpha=2$': 'WAIC$+2\sigma$ (Bayes)',
    'bayes_$\\alpha=2$': 'Bayes Loss+$2\sigma$',
    'waic_gibbs_$\\alpha=2$': 'WAIC$+2\sigma$ (Gibbs)',
    'gibbs_$\\alpha=2$': 'Gibbs Loss+$2\sigma$',
    'nlls_$\\beta=0.0$_$\\alpha=3$': 'MDL+$3\sigma$',
    'nlls_$\\beta=1.0$_$\\alpha=3$': '$\\beta=1.0$+$3\sigma$',
    'nlls_$\\beta=100000000.0$_$\\alpha=3$': 'CV$_{\\alpha=3}$',
    'variances_$\\alpha=3$': 'NLL Variance+$3\sigma$',
    'waic_bayes_$\\alpha=3$': 'WAIC$+3\sigma$ (Bayes)',
    'bayes_$\\alpha=3$': 'Bayes Loss+$3\sigma$',
    'waic_gibbs_$\\alpha=3$': 'WAIC$+3\sigma$ (Gibbs)',
    'gibbs_$\\alpha=3$': 'Gibbs Loss+$3\sigma$',
}
scale2name = {
    'abs': '',
    'diff': 'Gain in ',
    'rel': 'Relative ',
}


def plot_type2name(plot_type):
    return '-'.join(plot_type.split(' ')[:2])


def load_results(data_name, engines, num_trains, num_dev, seeds, rels, keys):
    num_train2engine2seed2rel2template2results = {}
    for num_train in tqdm(num_trains, desc='# Train'):
        num_train2engine2seed2rel2template2results[num_train] = {}
        for engine in tqdm(engines, desc='Engine'):
            num_train2engine2seed2rel2template2results[num_train][engine] = []
            assert (num_train > 0) or (num_dev is None), 'Not implemented using args.num_dev != None for few-shot evaluation'
            save_dir = f'{os.getenv("BASE")}/data/rel2template2results.data_name-{data_name}_UHN.engine-{engine}.num_train-{num_train}.sort_by_weight-False'
            if num_dev is not None:
                save_dir += f'.num_dev-{num_dev}'

            for seed in seeds:
                rel2template2results = {}
                for rel in rels:
                    rel2template2results[rel] = {}
                    for k in keys:
                        for template, value in np.load(f'{save_dir}/seed-{seed}/{rel if data_name == "TREx" else rel.lower()}.key-{k}.npz').items():
                            if template not in rel2template2results[rel]:
                                rel2template2results[rel][template] = {}
                            rel2template2results[rel][template]['nlls' if k == 'verbalizer_nlls' else k] = value
                num_train2engine2seed2rel2template2results[num_train][engine].append(rel2template2results)
    return num_train2engine2seed2rel2template2results


def compute_generalization_estimates(num_train2engine2seed2rel2template2results, data_name, engines, num_trains, num_dev, rels, keys, betas, norm_waic=False, tiebreak_eps=None):
    keys = ['nlls' if k == 'verbalizer_nlls' else k for k in keys]
    num_train2engine2seed2rel2stat2results = {}
    num_train2permutations = {num_train: generate_permutations(num_train, num_dev, data_name)[0] for num_train in num_trains}
    for num_train in tqdm(num_trains, desc='Computing Stats'):
        engine2seed2rel2stat2results = {}
        for engine in engines:
            seed2rel2stat2results = []
            for seed, rel2template2results in enumerate(num_train2engine2seed2rel2template2results[num_train][engine]):
                rel2stat2results = {}
                for rel in rels:
                    template2results = rel2template2results[rel]
                    template_list = list(sorted(template2results.keys()))
                    if len(template2results[template_list[0]]) <= 1:
                        continue
                    rel2stat2results[rel] = {}
                    # Group MDL/LOOCV/Test stats
                    for stat in keys:
                        rel2stat2results[rel][stat] = np.array([template2results[template][stat] for template in template_list])
                    if 'x0_nlls' in keys:
                        rel2stat2results[rel]['x0y_nlls'] = rel2stat2results[rel]['nlls'] + rel2stat2results[rel]['x0_nlls']
                    if 'x0y_nlls' in keys:
                        rel2stat2results[rel]['prompt_nlls'] = rel2stat2results[rel]['example_nlls'] - rel2stat2results[rel]['x0y_nlls']
                    if 'example_nlls' in keys:
                        rel2stat2results[rel]['nony_nlls'] = rel2stat2results[rel]['example_nlls'] - rel2stat2results[rel]['nlls']
                    if data_name == 'TREx':
                        ranks = np.array([template2results[template]['ranks'] for template in template_list])
                        accs = ranks == 0
                    else:
                        accs = np.array([template2results[template]['verbalizer_accs'] for template in template_list])
                    rel2stat2results[rel]['test_acc'] = accs[:, :, -1]
                    if tiebreak_eps is not None:  # NB: Can use e.g. 1e-6
                        rel2stat2results[rel]['mdl_acc'] = accs[:, :, :-1].mean(-1) - ((tiebreak_eps / 100.) * rel2stat2results[rel]['mdl'])
                        rel2stat2results[rel]['loo_acc'] = accs[:, :, -2] - ((tiebreak_eps / 100.) * rel2stat2results[rel]['loo'])
                    for beta in betas:
                        beta_name = f'$\\beta={beta}$'
                        weights = np.expand_dims(softmax(beta * np.arange(num_train)), [0, 1])
                        for stat in keys: # (['verbalizer_nlls', 'verbalizer_entropys'] if data_name != 'TREx' else []) + ['nlls', 'x0_nlls', 'example_nlls', 'entropys', 'x0y_nlls', 'prompt_nlls', 'nony_nlls']:
                            rel2stat2results[rel][f'{stat}_{beta_name}'] = (np.array(rel2stat2results[rel][stat])[:, :, :-1] * weights).sum(axis=2)
                            if stat in ['nlls'] and (tiebreak_eps is not None):
                                rel2stat2results[rel][f'{stat}_{beta_name}_acc'] = (accs[:, :, :-1] * weights).sum(-1) - ((tiebreak_eps / 100.) * rel2stat2results[rel][f'{stat}_{beta_name}'])
                            if stat in ['entropys', 'prompt_nlls', 'nony_nlls']:
                                rel2stat2results[rel][f'-{stat}_{beta_name}'] = -rel2stat2results[rel][f'{stat}_{beta_name}']
                    if 'entropys' in keys:
                        rel2stat2results[rel][f'entropys_test'] = np.array(rel2stat2results[rel]['entropys'])[:, :, -1]
                        rel2stat2results[rel][f'-entropys_test'] = -rel2stat2results[rel][f'entropys_test']
                    if 'nony_nlls' in keys:
                        rel2stat2results[rel]['nony_nlls_first'] = rel2stat2results[rel]['nony_nlls'][:, :, -1]
                        rel2stat2results[rel]['-nony_nlls_first'] = -rel2stat2results[rel]['nony_nlls_first']

                    # Compute WAICs
                    held_in_lls = (-rel2stat2results[rel]['nlls'][:, :len(num_train2permutations[num_train]), :num_train-1]).sum(2)
                    loo_sample = np.array(num_train2permutations[num_train])[:, -1]
                    perm_held_in_lls = np.array([held_in_lls[:, loo_sample == ex_no]
                                     for ex_no in range(num_train)]).transpose((1, 0, 2))
                    posterior = softmax(perm_held_in_lls, axis=2)
                    template2xy2lls = -np.array([rel2stat2results[rel]['nlls'][:, :len(num_train2permutations[num_train])][:, loo_sample == ex_no, num_train-1]
                                                 for ex_no in range(num_train)]).transpose((1, 0, 2))
                    template2xy2lls2 = -np.array([rel2stat2results[rel]['nlls_$\\beta=100000000.0$'][:, :len(num_train2permutations[num_train])][:, loo_sample == ex_no]
                                                 for ex_no in range(num_train)]).transpose((1, 0, 2))
                    assert np.sum(template2xy2lls != template2xy2lls2) == 0, f'Expected template2xy2lls == template2xy2lls2'

                    # Using computed posterior
                    bayes_loss = np.log((np.exp(template2xy2lls) * posterior).sum(2))
                    gibbs_loss = (template2xy2lls * posterior).sum(2)
                    variances = ((template2xy2lls ** 2) * posterior).sum(2) - (gibbs_loss ** 2)

                    waic1s = bayes_loss - (2 * (bayes_loss - gibbs_loss))

                    waics = bayes_loss - variances
                    waics_gibbs = gibbs_loss - variances
                    assert ((waics + 1e-8) >= waics_gibbs).all(), 'Expected Bayes WAIC >= Gibbs WAIC'
                    rel2stat2results[rel]['variances'] = np.array([variances[:, idx] for idx in loo_sample]).transpose((1, 0))
                    rel2stat2results[rel]['waic_bayes'] = np.array([waics[:, idx] for idx in loo_sample]).transpose((1, 0))
                    rel2stat2results[rel]['waic_gibbs'] = np.array([waics_gibbs[:, idx] for idx in loo_sample]).transpose((1, 0))
                    rel2stat2results[rel]['waic1'] = np.array([waic1s[:, idx] for idx in loo_sample]).transpose((1, 0))
                    rel2stat2results[rel]['bayes'] = -np.array([bayes_loss[:, idx] for idx in loo_sample]).transpose((1, 0))
                    rel2stat2results[rel]['gibbs'] = -np.array([gibbs_loss[:, idx] for idx in loo_sample]).transpose((1, 0))

                seed2rel2stat2results.append(rel2stat2results)

            engine2seed2rel2stat2results[engine] = seed2rel2stat2results
        num_train2engine2seed2rel2stat2results[num_train] = engine2seed2rel2stat2results
    return num_train2engine2seed2rel2stat2results


def compute_stats(num_train2engine2seed2rel2stat2results, num_train2num_samples, num_train2cyclic_permutations, data_name, engines, num_trains, rels, plot_stat_names, plot_types, num_total_dev, test_transfer=False):
    plot_type2num_train2engine2engine2stat2results = {}
    for plot_type in tqdm(plot_types, desc='Plot Type'):
        num_train2engine2engine2stat2results = {}
        for num_train in tqdm(num_trains, desc='# Train'):
            num_train2engine2engine2stat2results[num_train] = {}
            for engine2 in tqdm(engines, desc='Engine'):  # Model which we measure accuracy with
                for engine1 in (engines if test_transfer else [engine2]):  # Model which we choose prompt with
                    if engine1 not in num_train2engine2engine2stat2results[num_train]:
                        num_train2engine2engine2stat2results[num_train][engine1] = {}

                    scale = 100 if '(%)' in plot_type else 1
                    stat2 = 'test'
                    best_func = np.min
                    if plot_type in {'Test Accuracy of Chosen Prompt (%)', 'Accuracy at Choosing Best Prompt (%)'}:
                        stat2 = 'test_acc'
                        best_func = np.max
                    elif plot_type in {'Frequency of Choosing Worst Prompt (%)'}:
                        stat2 = 'test_acc'
                        best_func = np.min
                    elif plot_type == 'Test MRR of Chosen Prompt (%)':
                        stat2 = 'test_mrr'
                        best_func = np.max

                    # Evaluate baselines
                    if plot_type in ['Test Accuracy of Chosen Prompt (%)', 'Test MRR of Chosen Prompt (%)', 'Test Loss of Chosen Prompt']:
                        seed2best = []
                        seed2mean = []
                        seed2worst = []
                        for seed, rel2stat2results in enumerate(num_train2engine2seed2rel2stat2results[num_train][engine2]):
                            test = [rel2stat2results[rel][stat2].mean(axis=1) for rel in rels]
                            seed2best.append(np.mean([best_func(t) for t in test]))
                            seed2worst.append(np.mean([best_func(-t) for t in test]))
                            seed2mean.append(np.mean([t.mean() for t in test]))
                        seed2best = np.array(seed2best) * scale
                        seed2worst = np.array(seed2worst) * scale
                        seed2mean = np.array(seed2mean) * scale
                    else:
                        seed2best = None
                        seed2worst = None
                        seed2mean = None

                    num_train2engine2engine2stat2results[num_train][engine1][engine2] = {}
                    for stat1 in plot_stat_names:
                        # Get results
                        for alpha in [0, 1, 2, 3]:
                            stat_type2num_sample2seed2rel2scores = {'abs': {}, 'diff': {}, 'rel': {}}
                            for num_sample in num_train2num_samples[num_train]:
                                seed2rel2sample2score = []
                                for seed, rel2stat2results2 in enumerate(num_train2engine2seed2rel2stat2results[num_train][engine2]):
                                    rel2sample2score = []
                                    for rel in rels:
                                        if rel[0] != 'P':
                                            assert len(rels), f'Only LAMA tasks can be grouped together. Please use a single element list for "rels".'
                                        stat2results1 = num_train2engine2seed2rel2stat2results[num_train][engine1][seed][rel]
                                        stat2results2 = rel2stat2results2[rel]
                                        full_dev_eval_end = None
                                        assert not ((rel.lower() in {'cb', 'copa', 'wsc'}) and (num_total_dev is None)), f'Please set num_total_dev for {rel}'
                                        if (num_total_dev is not None) and (num_total_dev < max(num_train2num_samples[num_train])):
                                            full_dev_eval_end = int((max(num_train2num_samples[num_train]) // num_total_dev) * num_total_dev)
                                        estat2 = stat2results2[stat2][:full_dev_eval_end].mean(axis=1)

                                        sample2score = []
                                        for i in range(max(num_train2num_samples[num_train]) // num_sample):
                                            subsampled_stats = stat2results1[stat1][:, list(range(len(num_train2cyclic_permutations[num_train]))) if data_name != 'TREx' else num_train2cyclic_permutations[num_train]][:, i * num_sample: (i + 1) * num_sample]
                                            estat1 = subsampled_stats.mean(axis=1)
                                            if (alpha != 0) and (num_sample > 1):
                                                estat1 = estat1 + (statsign(stat1) * alpha * subsampled_stats.std(axis=1, ddof=1))
                                            if plot_type == 'Pairwise Prompt Ranking Accuracy (%)':
                                                score = (kendalltau(statsign(stat2) * estat2, statsign(stat1) * estat1).correlation + 1) / 2.
                                            elif plot_type == 'Accuracy at Choosing Best Prompt (%)':
                                                score = np.min(statsign(stat2) * estat2) == (statsign(stat2) * estat2[np.argmin(statsign(stat1) * estat1)])
                                            elif plot_type == 'Frequency of Choosing Worst Prompt (%)':
                                                score = np.max(statsign(stat2) * estat2) == (statsign(stat2) * estat2[np.argmin(statsign(stat1) * estat1)])
                                            elif plot_type in ['Test Accuracy of Chosen Prompt (%)', 'Test MRR of Chosen Prompt (%)', 'Test Loss of Chosen Prompt']:
                                                score = estat2[np.argmin(statsign(stat1) * estat1)]
                                            else:
                                                NotImplemented(f'plot_type = {plot_type}')
                                            sample2score.append(score * scale)
                                        rel2sample2score.append(sample2score)
                                    seed2rel2sample2score.append(rel2sample2score)
                                seed2rel2sample2score = np.array(seed2rel2sample2score)
                                stat_type2num_sample2seed2rel2scores['abs'][num_sample] = seed2rel2sample2score
                                if seed2mean is not None:
                                    stat_type2num_sample2seed2rel2scores['diff'][num_sample] = seed2rel2sample2score - np.expand_dims(seed2mean, axis=(1,2))
                                    stat_type2num_sample2seed2rel2scores['rel'][num_sample] = 100. * stat_type2num_sample2seed2rel2scores['diff'][num_sample] / np.expand_dims(seed2best - seed2mean, axis=(1,2))

                            for stat_type, num_sample2seed2rel2scores in stat_type2num_sample2seed2rel2scores.items():
                                alpha_str = "" if alpha == 0 else f"_$\\alpha={alpha}$"
                                num_train2engine2engine2stat2results[num_train][engine1][engine2][f'{stat_type}_{stat1}{alpha_str}'] = num_sample2seed2rel2scores

                    if seed2mean is not None:
                        for score_name, scores in zip(['mean', 'best', 'worst'], [seed2mean, seed2best, seed2worst]):
                            num_train2engine2engine2stat2results[num_train][engine1][engine2][f'abs_{score_name}'] = scores
                            num_train2engine2engine2stat2results[num_train][engine1][engine2][f'diff_{score_name}'] = scores - seed2mean
                            num_train2engine2engine2stat2results[num_train][engine1][engine2][f'rel_{score_name}'] = 100. * num_train2engine2engine2stat2results[num_train][engine1][engine2][f'diff_{score_name}'] / (seed2best - seed2mean)
        plot_type2num_train2engine2engine2stat2results[plot_type] = num_train2engine2engine2stat2results
    return plot_type2num_train2engine2engine2stat2results


def plot_compute_efficiency(plot_type2num_train2engine2engine2stat2results, plot_type, num_train, engine1, engine2, plot_stat_names, num_train2num_samples, scale, plot_num_estimates, plot_data_name, show_legend=True, show_y=True, notebook=False, save_dir=None):
    fig, ax = plt.subplots()
    xmax = max(num_train2num_samples[num_train])
    if plot_num_estimates and ('beta=0.0' not in plot_stat_names):
        xmax //= num_train
    plt.xlim(xmin=1, xmax=xmax)

    # Plot results
    for stat_no, stat in enumerate(plot_stat_names):
        num_sample2seed2rel2scores = plot_type2num_train2engine2engine2stat2results[plot_type][num_train][engine1][engine2][f'{scale}_{stat}']
        means = []
        stderrs = []
        for num_sample in num_train2num_samples[num_train]:
            mean_mc_scores = num_sample2seed2rel2scores[num_sample].mean((1, 2))
            assert len(mean_mc_scores.shape) == 1, f'Expected len(mean_mc_scores.shape) ({len(mean_mc_scores.shape)}) == 1'
            means.append(np.mean(mean_mc_scores))
            stderrs.append(np.std(mean_mc_scores, ddof=1))
        means = np.array(means)
        stderrs = np.array(stderrs)

        if plot_num_estimates and ('beta=0.0' not in stat):
            means = means[num_train2num_samples[num_train] >= num_train]
            stderrs = stderrs[num_train2num_samples[num_train] >= num_train]
            xs = num_train2num_samples[num_train][num_train2num_samples[num_train] >= num_train] / float(num_train)
        else:
            xs = num_train2num_samples[num_train]
        plt.plot(xs, means, label=plot_stat2name[stat], color=cm(stat_no / len(plot_stat_names)))
        plt.fill_between(xs, means - stderrs, means + stderrs, alpha=0.3, linewidth=0, color=cm(stat_no / len(plot_stat_names)))

    # Plot baselines
    if plot_type in ['Test Accuracy of Chosen Prompt (%)', 'Test MRR of Chosen Prompt (%)', 'Test Loss of Chosen Prompt']:
        for stat, color in zip(['best', 'mean'], ['k', 'gray']):
            scores = plot_type2num_train2engine2engine2stat2results[plot_type][num_train][engine1][engine2][f'{scale}_{stat}']
            plt.plot([1, xmax], [scores.mean()] * 2, label=plot_stat2name[stat], color=color)
            plt.fill_between([1, xmax], [scores.mean() - np.std(scores, ddof=1)] * 2, [scores.mean() + np.std(scores, ddof=1)] * 2, color='gray', alpha=0.3, linewidth=0)

    # Draw plot
    if show_legend:
        plt.legend(title='Selection\nMethod', loc='upper left', fontsize=12, title_fontsize=12, bbox_to_anchor=(0, 0.95))
    print(f'{plot_data_name}: {engine2name[engine2]}')
    plt.title(f'$\\bf{{{num_train}-Shot}}$', fontsize=15)
    if show_y:
        plt.ylabel((scale2name[scale] + plot_type).replace('Test ', '').replace('Choosing Best', 'Choosing\nBest').replace('Accuracy of', 'Accuracy\nof'), fontsize=15)
        plt.yticks(fontsize=14)
    else:
        ax.set_yticklabels([])
    plt.xlabel('# Estimates' if plot_num_estimates else 'Number of Forward Passes', fontsize=16)
    plt.xscale('log')
    ax.set_xticks([], minor=True)
    ax.set_xticks(num_train2num_samples[num_train])
    ax.set_xticklabels(num_train2num_samples[num_train], fontsize=14)
    plt.grid()
    if save_dir is not None:
        save_file = f'{save_dir}/{inspect.stack()[0][3]}.plot_type-{plot_type2name(plot_type)}.engine-{engine2}.num_train-{num_train}.pdf'
        plt.savefig(save_file, bbox_inches='tight')
        print('Saving to:', save_file)
    if notebook:
        plt.show()
    plt.clf()


def plot_prompt_transfer(plot_type2num_train2engine2engine2stat2results, plot_type, num_train, engines, stat, num_train2num_samples, scale, num_samples, plot_data_name, show_y=True, show_cbar=True, notebook=False, save_dir=None):
    assert num_samples in {'One', 'Multi'}, f'Expected num_samples ({num_samples}) in ["One", "Multi"]'
    if len(engines) <= 1:
        return
    engine_names = [engine2name[engine] for engine in engines]
    num_samples_index = -1 if num_samples == 'Multi' else (0 if 'beta=0.0' in stat else 1)
    data = np.array([[np.mean(plot_type2num_train2engine2engine2stat2results[plot_type][num_train][engine1][engine2][f'{scale}_{stat}'][num_train2num_samples[num_train][num_samples_index]].mean((1, 2))) for engine2 in engines] for engine1 in engines]).transpose()
    print('Fixed model we choose prompts with:', data[:, 0])
    print('Fixed model we eval acc with:      ', data[0])
    fig, ax = plt.subplots()
    im, cbar = heatmap(data, engine_names, engine_names, ax=ax, cmap="Blues", cbarlabel=f"{scale2name[scale]}Test Accuracy of Chosen Prompt(%)".replace('Accuracy of', 'Accuracy\nof') if show_cbar else None, vmin=0, vmax=55)
    annotate_heatmap(im, valfmt="{x:." + str(0 if len(engines) > 6 else 1) + "f}")
    plt.xlabel(f'Prompt Selection Model', fontsize=16)
    plt.xticks(fontsize=14)
    if show_y:
        plt.ylabel('Prediction Model', fontsize=16)
        plt.yticks(fontsize=14)
    else:
        ax.set_yticklabels([])
        ax.set_yticks([])
    print(f'{plot_data_name}: Prompt Transfer across Models\n {plot_stat2name[stat]} ({num_train}-shot{"" if num_samples == "Multi" else ", One Sample Estimate"})')
    plt.title(f'{plot_stat2name[stat]}', fontsize=16, fontweight='bold')
    fig.tight_layout()
    if save_dir is not None:
        save_file = f'{save_dir}/{inspect.stack()[0][3]}.plot_type-{plot_type2name(plot_type)}.stat-{stat}.num_train-{num_train}.pdf'
        plt.savefig(save_file, bbox_inches='tight')
        print('Saving to:', save_file)
    if notebook:
        plt.show()
    plt.clf()


def plot_results_by_engine(acc_plot_type2num_train2engine2engine2stat2results, plot_type, num_train, engines, plot_stat_names, acc_num_train2num_samples, scale, num_samples, top, bottom, plot_data_name, show_legend=True, show_y=True, show_legend_title=True, legend_bbox_to_anchor=None, figsize=None, notebook=False, save_dir=None):
    assert num_samples in {'One', 'Multi'}, f'Expected num_samples ({num_samples}) in ["One", "Multi"]'
    plot_baselines = (plot_type == 'Test Accuracy of Chosen Prompt (%)') and (scale == 'abs')
    num_bars = (len(plot_stat_names) + (3 if plot_baselines else 0))
    width = (1. / (num_bars + 1))
    x = np.arange(len(engines))  # the label locations
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot means
    if plot_baselines:
        for baseline, color in zip(['worst', 'mean'], ['r', 'orange' if figsize is None else 'gray']):
            means = []
            stderrs = []
            for engine2 in engines:
                engine1 = engine2
                mean_mc_scores = acc_plot_type2num_train2engine2engine2stat2results[plot_type][num_train][engine1][engine2][f'{scale}_{baseline}']
                means.append(np.mean(mean_mc_scores))
                stderrs.append(sem(mean_mc_scores, axis=None))
            if baseline == 'worst':
                means = -np.array(means)
                print(means)
            ax.bar(x, means, width, yerr=stderrs, label=plot_stat2name[baseline], color=color)
            x = x + width
        
    # Plot prompt selection results
    for stat_no, stat in enumerate(plot_stat_names):
        num_samples_index = -1 if num_samples == 'Multi' else (0 if 'beta=0.0' in stat else 1)
        means = []
        stderrs = []
        for engine2 in engines:
            engine1 = engine2
            num_sample2seed2rel2scores = acc_plot_type2num_train2engine2engine2stat2results[plot_type][num_train][engine1][engine2][f'{scale}_{stat}']
            mean_mc_scores = num_sample2seed2rel2scores[acc_num_train2num_samples[num_train][num_samples_index]]
            means.append(np.mean(mean_mc_scores))
            stderrs.append(sem(mean_mc_scores, axis=None))
        ax.bar(x, means, width, yerr=stderrs, label=plot_stat2name[stat], color=cm(stat_no / len(plot_stat_names)))
        x = x + width
    
    # Plot best prompt
    if plot_baselines:
        means = []
        stderrs = []
        for engine2 in engines:
            engine1 = engine2
            mean_mc_scores = acc_plot_type2num_train2engine2engine2stat2results[plot_type][num_train][engine1][engine2][f'{scale}_best']
            means.append(np.mean(mean_mc_scores))
            stderrs.append(sem(mean_mc_scores, axis=None))
        ax.bar(x, means, width, yerr=stderrs, label=plot_stat2name['best'], color='g')
        x = x + width

    plt.ylim(top=top, bottom=bottom)
    if show_y:
        ylabel = (scale2name[scale] + plot_type).replace('Accuracy', 'Acc.').replace('Choosing Worst', 'Choosing\nWorst').replace('Best', 'Top')
        if scale == 'rel':
            ylabel = ylabel.replace('of Chosen', 'of\nChosen')
        ax.set_ylabel(ylabel, fontsize=15)
    plt.yticks(fontsize=14)
    ax.set_xticks(x - ((num_bars + 1) * width / 2))
    ax.set_xticklabels([engine2name[engine] for engine in engines])
    plt.xticks(fontsize=14)
    ax.set_xlabel('Model Parameters (B)', fontsize=16)
    print(f'({num_train}-shot{"" if num_samples == "Multi" else ", One Sample Estimate"})')
    if plot_data_name != 'LAMA-UHN':
        plt.title(f'$\\bf{{{plot_data_name}}}$', fontsize=16) # 18
    plt.xlim(left=-width, right=x[-1])
    ax.set_axisbelow(True)
    plt.grid(axis='y')
    fig_width, fig_height = plt.gcf().get_size_inches()
    print(fig_width, fig_height)
    if show_legend:
        if show_legend_title:
            plt.legend(title='Selection\nMethod', fontsize=12, title_fontsize=12, bbox_to_anchor=legend_bbox_to_anchor) # loc='upper left', 
        else:
            plt.legend(fontsize=12, bbox_to_anchor=legend_bbox_to_anchor) # loc='upper left', 
    if save_dir is not None:
        save_file = f'{save_dir}/{inspect.stack()[0][3]}.plot_type-{plot_type2name(plot_type)}.num_samples-{num_samples}.num_train-{num_train}.scale-{scale}.pdf'
        plt.savefig(save_file, bbox_inches='tight')
        print('Saving to:', save_file)
    if notebook:
        plt.show()
    plt.clf()


def plot_results_by_num_train(plot_type2num_train2engine2engine2stat2results, plot_type, num_trains, engines, stat, num_train2num_samples, scale, num_samples, show_legend=True, show_y=True, notebook=False, save_dir=None):
    assert num_samples in {'One', 'Multi'}, f'Expected num_samples ({num_samples}) in ["One", "Multi"]'
    fig, ax = plt.subplots()
    plt.xlim(left=min(num_trains), right=max(num_trains))
    for engine_no, engine in enumerate(engines):
        num_samples_index = -1 if num_samples == 'Multi' else (0 if 'beta=0.0' in stat else 1)
        means = []
        stderrs = []
        for num_train in num_trains:
            num_sample2seed2rel2scores = plot_type2num_train2engine2engine2stat2results[plot_type][num_train][engine][engine][f'{scale}_{stat}']
            mean_mc_scores = num_sample2seed2rel2scores[num_train2num_samples[num_train][num_samples_index]].mean((1, 2))
            means.append(np.mean(mean_mc_scores))
            stderrs.append(sem(mean_mc_scores, axis=None))
        means = np.array(means)
        stderrs = np.array(stderrs)

        plt.plot(num_trains, means, label=engine2name[engine], color=cm(engine_no / len(engines)))
        plt.fill_between(num_trains, means - stderrs, means + stderrs, alpha=0.3, linewidth=0, color=cm(engine_no / len(engines)))

    if show_y:
        plt.ylabel((scale2name[scale] + plot_type).replace('Test ', '').replace('Choosing Best', 'Choosing\nBest').replace('of Chosen', 'of\nChosen'), fontsize=15)
        plt.yticks(fontsize=14)
    else:
        ax.set_yticklabels([])
    if show_legend:
        plt.legend(title='Model', bbox_to_anchor=(1, 1), fontsize=13, title_fontsize=15)
    plt.title(f'$\\bf{{{plot_stat2name[stat]}}}${"" if num_samples == "Multi" else " (One Sample Estimate)"}', fontsize=15)
    plt.xlabel('Number of Examples', fontsize=16)
    ax.set_xticks(num_trains)
    plt.xticks(fontsize=14)
    plt.grid()
    if save_dir is not None:
        save_file = f'{save_dir}/{inspect.stack()[0][3]}.plot_type-{plot_type2name(plot_type)}.stat-{stat}.num_samples-{num_samples}.pdf'
        plt.savefig(save_file, bbox_inches='tight')
        print('Saving to:', save_file)
    if notebook:
        plt.show()
    plt.clf()


def plot_results_distribution(plot_type2num_train2engine2engine2stat2results, plot_type, num_train, engines, plot_stat_names, num_train2num_samples, scale, num_samples, plot_info, plot_data_name, show_legend=True, show_y=True, notebook=False, save_dir=None):
    assert num_samples in {'One', 'Multi'}, f'Expected num_samples ({num_samples}) in ["One", "Multi"]'
    save_filebase = f'{save_dir}/{inspect.stack()[0][3]}.plot_type-{plot_type2name(plot_type)}.num_train-{num_train}.plot_info-{plot_info}'
    gains = []
    if plot_info == 'expectation':
        width = 1. / (len(plot_stat_names) + 1)
        x = np.arange(len(engines))  # the label locations
        fig, ax = plt.subplots()

    for stat_no, stat in enumerate(plot_stat_names):
        num_samples_index = -1 if num_samples == 'Multi' else (0 if 'beta=0.0' in stat else 1)
        if plot_info == 'cdf':
            fig, ax = plt.subplots()

        means = []
        stddevs = []
        for engine_no, engine in enumerate(engines):
            num_sample2seed2rel2scores = plot_type2num_train2engine2engine2stat2results[plot_type][num_train][engine][engine][f'{scale}_{stat}']
            valid_stats = num_sample2seed2rel2scores[num_train2num_samples[num_train][num_samples_index]].flatten()
            gains.append((plot_stat2name[stat], engine2name[engine], valid_stats.mean(), valid_stats.std(ddof=1)))
            means.append(valid_stats.mean())
            stddevs.append(valid_stats.std(ddof=1))

            if plot_info == 'pdf':
                plt.hist(valid_stats)
                print(f'({num_train}-shot{"" if num_samples == "Multi" else ", One Sample Estimate"})')
                plt.title(f'$\\bf{{{plot_stat2name[stat]}}}$', fontsize=15)
                if plot_data_name == 'TREx':
                    plt.tight_layout()
                if save_dir is not None:
                    save_file = f'{save_filebase}.stat-{stat}.engine-{engine}.pdf'
                    plt.savefig(save_file, bbox_inches='tight')
                    print('Saving to:', save_file)
                if notebook:
                    plt.show()
                plt.clf()
            elif plot_info == 'cdf':
                plt.plot(sorted(valid_stats), np.linspace(0, 100, len(valid_stats)), label=engine2name[engine], color=cm(engine_no / len(engines)))
        if plot_info == 'cdf':
            if show_legend:
                plt.legend(title='Params (B)', fontsize=12, title_fontsize=11)
            plt.ylim(bottom=0, top=100)
            if show_y:
                plt.ylabel('% of Time Acc. Gain\nBelow Threshold', fontsize=16)
                plt.yticks(fontsize=14)
            else:
                ax.set_yticklabels([])
            plt.xticks(fontsize=14)
            plt.xlabel('Threshold for Acc. Gain over Mean Prompt', fontsize=16)
            print(f'Chance of Accuracy Gain with {plot_stat2name[stat]} ({num_train}-shot{"" if num_samples == "Multi" else ", One Sample Estimate"})')
            title = f'$\\bf{{{plot_data_name if plot_data_name != "TREx" else plot_stat2name[stat]}}}$'
            print(title)
            if plot_data_name != 'LAMA-UHN':
                plt.title(title, fontsize=18)
            plt.grid()
            if plot_data_name == 'TREx':
                plt.tight_layout()
            if save_dir is not None:
                save_file = f'{save_filebase}.stat-{stat}.pdf'
                plt.savefig(save_file, bbox_inches='tight')
                print('Saving to:', save_file)
            if notebook:
                plt.show()
            plt.clf()
        elif plot_info == 'expectation':
            ax.bar(x, means, width, yerr=stddevs, label=plot_stat2name[stat], color=cm(stat_no / len(plot_stat_names)))
            x = x + width

    if plot_info == 'expectation':
        if show_y:
            ax.set_ylabel('Accuracy Gain (%)', fontsize=15)
            plt.yticks(fontsize=14)
        else:
            ax.set_yticklabels([])
        ax.set_xticks(x - ((len(plot_stat_names) + 1) * width / 2))
        ax.set_xticklabels([engine2name[engine] for engine in engines])
        plt.xticks(rotation=0)
        print(f'({num_train}-shot{"" if num_samples == "Multi" else ", One Sample Estimate"})')
        plt.title(f'$\\bf{{{plot_data_name}}}$', fontsize=15)
        plt.xlim(left=-width, right=x[-1])
        if show_legend:
            ax.legend(bbox_to_anchor=(1, 0.9), fontsize=12)
        plt.grid(axis='y')
        if plot_data_name == 'TREx':
            fig.tight_layout()
        if save_dir is not None:
            save_file = f'{save_filebase}.pdf'
            plt.savefig(save_file, bbox_inches='tight')
            print('Saving to:', save_file)
        if notebook:
            plt.show()
        plt.clf()

    return pd.DataFrame(gains, columns=['Criterion', 'Model', '$\mu$', '$\sigma$'])


def plot_results_distribution_by_stat(plot_type2num_train2engine2engine2stat2results, plot_type, num_train, engines, plot_stat_names, num_train2num_samples, scale, num_samples, plot_info, plot_data_name, show_y=True, notebook=False, save_dir=None):
    save_filebase = f'{save_dir}/{inspect.stack()[0][3]}.plot_type-{plot_type2name(plot_type)}.num_train-{num_train}.plot_info-{plot_info}'
    cm = plt.cm.plasma
    assert num_samples in {'One', 'Multi'}, f'Expected num_samples ({num_samples}) in ["One", "Multi"]'
    gains = []
    if plot_info == 'expectation':
        width = 1. / (len(engines) + 1)
        x = np.arange(len(plot_stat_names))  # the label locations
        fig, ax = plt.subplots()

    for engine_no, engine in enumerate(engines):
        if plot_info == 'cdf':
            fig, ax = plt.subplots()
        
        means = []
        stddevs = []
        for stat_no, stat in enumerate(plot_stat_names):
            num_samples_index = -1 if num_samples == 'Multi' else (0 if 'beta=0.0' in stat else 1)
            num_sample2seed2rel2scores = plot_type2num_train2engine2engine2stat2results[plot_type][num_train][engine][engine][f'{scale}_{stat}']
            valid_stats = num_sample2seed2rel2scores[num_train2num_samples[num_train][num_samples_index]].flatten()
            gains.append((plot_stat2name[stat], engine2name[engine], valid_stats.mean(), valid_stats.std(ddof=1)))
            means.append(valid_stats.mean())
            stddevs.append(valid_stats.std(ddof=1))

            if plot_info == 'pdf':
                plt.hist(valid_stats)
                title = f'{plot_stat2name[stat]} {num_samples} Sample Estimate ({engine2name[engine2]} {num_train}-shot)'
                plt.title(title)
                if save_dir is not None:
                    save_file = f'{save_filebase}.engine-{engine}.stat-{plot_stat2name[stat]}.pdf'
                    plt.savefig(save_file, bbox_inches='tight')
                    print('Saving to:', save_file)
                if notebook:
                    plt.show()
                plt.clf()
            elif plot_info == 'cdf':
                plt.plot(sorted(valid_stats), np.linspace(0, 100, len(valid_stats)), label=plot_stat2name[stat], color=cm(stat_no / len(plot_stat_names)))
        if plot_info == 'cdf':
            plt.legend(fontsize=15)
            plt.ylim(bottom=0, top=100)
            if show_y:
                plt.ylabel('% of Time Acc. Gain\nBelow Threshold', fontsize=16)
                plt.yticks(fontsize=14)
            else:
                ax.set_yticklabels([])
            plt.yticks(fontsize=14)
            plt.xticks(fontsize=14)
            plt.xlabel('Threshold for Acc. Gain over Random Prompt', fontsize=15)
            title = f'UCB: Chance of Acc. Gain (GPT3 {engine2name[engine]}B{"" if num_samples == "Multi" else ", One Sample Estimate"})'
            print(title)
            if plot_data_name != 'LAMA-UHN':
                plt.title(title, fontsize=18)
            plt.grid()
#             plt.tight_layout()
            if save_dir is not None:
                save_file = f'{save_filebase}.engine-{engine}.pdf'
                plt.savefig(save_file, bbox_inches='tight')
                print('Saving to:', save_file)
            if notebook:
                plt.show()
            plt.clf()
        elif plot_info == 'expectation':
            ax.bar(x, means, width, yerr=stddevs, label=plot_stat2name[stat], color=cm(stat_no / len(plot_stat_names)))
            x = x + width

    if plot_info == 'expectation':
        if show_y:
            ax.set_ylabel('Accuracy Gain (%)', fontsize=15)
            plt.yticks(fontsize=14)
        else:
            ax.set_yticklabels([])
        ax.set_xticks(x - ((len(engines) + 1) * width / 2))
        ax.set_xticklabels([plot_stat2name[stat] for stat in plot_stat_names])
        plt.xticks(rotation=0)
        plt.title(f'{num_samples} Sample Estimate')
        plt.xlim(left=-width, right=x[-1])
        ax.legend(bbox_to_anchor=(1, 0.9))
        plt.grid(axis='y')
        fig.tight_layout()
        if save_dir is not None:
            save_file = f'{save_filebase}.pdf'
            plt.savefig(save_file, bbox_inches='tight')
            print('Saving to:', save_file)
        if notebook:
            plt.show()
        plt.clf()

    return pd.DataFrame(gains, columns=['Criterion', 'Model', '$\mu$', '$\sigma$'])
