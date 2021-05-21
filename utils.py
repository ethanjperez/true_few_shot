import itertools
import json
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import openai
import random
from scipy.special import logsumexp, softmax, log_softmax
import string
import time
from tqdm.auto import tqdm, trange

# Only import torch if available (i.e., when running GPT2 models)
try:
    import torch
except ModuleNotFoundError:
    pass

lm_vocab_size = 50257

engine2name = {
    'ada': '2.7',
    'babbage': '6.7',
    'curie': '13',
    'davinci': '175',
    'distilgpt2': '0.08',
    'gpt2': '0.1',
    'gpt2-medium': '0.3',
    'gpt2-large': '0.8',
    'gpt2-xl': '1.5',
    'EleutherAI/gpt-neo-125M': '0.1 (Neo)',
    'EleutherAI/gpt-neo-1.3B': '1.3 (Neo)',
    'EleutherAI/gpt-neo-2.7B': '2.7 (Neo)',
}

engine2bs = { # in terms of number of training examples
    'ada': 600,
    'babbage': 600,
    'curie': 600,
    'davinci': 600,
    'distilgpt2': 2400,
    'gpt2': 2400,
    'gpt2-medium': 1200,
    'gpt2-large': 600,
    'gpt2-xl': 600,
    'EleutherAI/gpt-neo-125M': 600,
    'EleutherAI/gpt-neo-1.3B': 600,
    'EleutherAI/gpt-neo-2.7B': 300,
}


def read_jsonl(filepath):
    with open(filepath) as f:
        data = [json.loads(line) for line in f]
    return data


def normalize(text):
    for s in ['-']:
        text = text.replace(f' {s} ', s)
    for s in ['.', ',', ';', ':', '!', '?', ')']:
        text = text.replace(f' {s}', s)
    for s in ['(']:
        text = text.replace(f'{s} ', s)
    for s in ['[X]', '[Y]']:
        text = text.replace(s, f' {s}').replace(f'  {s}', f' {s}')
    return text.strip(' ')


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))


def statsign(stat):
    return -1 if (('acc' in stat) or ('mrr' in stat) or ('waic' in stat)) else 1


def preprocess_wsc(data):
    data_dict = {}
    for split in ['train', 'validation']:
        data_df = dict(data[split].to_pandas().transpose())
        data_dict[split] = [data_df[i] for i in range(len(data_df))]
        for d_no, d in enumerate(data[split]):
            words = d['text'].split(' ')
            for i in [1, 2]:
                span_start_index = d[f'span{i}_index']
                if (split, d_no, i) == ('validation', 42, 2):
                    span_start_index += 1
                char_span_start_index = len(' '.join(words[:span_start_index]))
                if char_span_start_index > 0:
                    char_span_start_index += 1
                char_span_end_index = char_span_start_index + len(d[f'span{i}_text'])
                assert d['text'][char_span_start_index: char_span_end_index] == d[f'span{i}_text'], f"{d['text'][char_span_start_index: char_span_end_index]} != {d[f'span{i}_text']}"
                if i == 2:
                    data_dict[split][d_no]['text_highlighted'] = '*'.join([d['text'][:char_span_start_index], d['text'][char_span_start_index: char_span_end_index], d['text'][char_span_end_index:]])
        if split == 'train':
            data_dict[split] = [d for d in data_dict[split] if d['label'] == 1]
    return data_dict


def get_logprobs(prompt, engine='ada', retry_delay=10, logprobs=100, model=None, tokenizer=None, batch_size=4):
    if model is not None:
        assert tokenizer is not None, 'Expected tokenizer alongside model'
        return get_model_response(model, tokenizer, prompt, num_log_probs=logprobs, batch_size=batch_size)
    else:
        all_logprobs = []
        for prompt_chunk in tqdm(chunks(prompt, batch_size), total=math.ceil(len(prompt) / batch_size)):
            sleep_time = 0
            while True:
                try:
                    response = openai.Completion.create(engine=engine, prompt=prompt_chunk, max_tokens=0, echo=True, logprobs=logprobs)
                    break
                except (openai.APIError, openai.error.RateLimitError, openai.error.APIConnectionError) as e:
                    if sleep_time == 0:
                        print('Sleeping...')
                    sleep_time += retry_delay
                    time.sleep(retry_delay)
            if sleep_time != 0:
                print(f'\tSlept {sleep_time}s')
            all_logprobs += [c['logprobs'] for c in response['choices']]
        return all_logprobs if (len(all_logprobs) > 0) else all_logprobs[0]


def format_prompt(train, d, last_train_idx, permutation, verbalizer=[], rel=None, instruction='\n', template=''):
    examples = [train[tidx] for tidx in permutation] + [d]
    char_spans = []
    
    template = template.strip()
    assert template.count('[Y]') == 1, template
    if len(examples) == 1:
        template = template.strip('. ')
    if rel in {'boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc.fixed'}:
        template = template.replace('[Y]', f'[X{len(examples[0]) - 1}]')
        x_no2loc = {}
        for x_no in range(len(examples[0])):
            x_str = f'[X{x_no}]'
            assert template.count(x_str) == 1, f'template.count({x_str}) ({template.count(x_str)}) != 1'
            assert x_str not in instruction, f'{x_str} not in instruction: {instruction}'
            x_no2loc[x_no] = template.index(f' {x_str}') if f' {x_str}' in template else template.index(x_str)
        x_nos = sorted(x_no2loc, key=x_no2loc.get)
#         print(template, '||', x_nos)

        prompt = instruction
        for ex in examples:
            xs = [verbalizer[x] if x_no == (len(ex) - 1) else x.strip() for x_no, x in enumerate(ex)]
            if rel in {'cb', 'rte'}:
                xs[1] = xs[1].rstrip('.?!')
            
            char_span_start = {'example': len(prompt)}
            char_span_end = {}
            example_text = template
            for x_no_idx, x_no in enumerate(x_nos):
                char_span_start[f'x{x_no}'] = len(prompt) + x_no2loc[x_no]
                for x_no2 in x_nos[:x_no_idx]:
                    char_span_start[f'x{x_no}'] += (len(xs[x_no2]) - len(f'[X{x_no2}]'))
                char_span_end[f'x{x_no}'] = char_span_start[f'x{x_no}'] + len(xs[x_no])
                example_text = example_text.replace(f'[X{x_no}]', xs[x_no])
                if example_text[char_span_start[f'x{x_no}'] - len(prompt)] == ' ':
                    char_span_end[f'x{x_no}'] += 1
            prompt += example_text

            for x_no in x_nos:
                assert prompt[char_span_start[f'x{x_no}']: char_span_end[f'x{x_no}']].strip() == xs[x_no].strip(), f"'{prompt[char_span_start[f'x{x_no}']: char_span_end[f'x{x_no}']]}' != '{xs[x_no]}'"
            char_span_start['y'] = char_span_start.pop(f'x{len(ex) - 1}')
            char_span_end['y'] = char_span_end.pop(f'x{len(ex) - 1}')
            if len(examples) > 1:
                prompt += '\n\n' if ('\n' in template) else '\n'
            else:
                assert len(prompt) == char_span_end['y'], f'len(prompt) ({len(prompt)}) != char_span_end["y"] ({char_span_end["y"]})'

            char_span_end['example'] = len(prompt)
            char_spans.append([char_span_start, char_span_end])
    elif (rel[0] == 'P') and (rel[1:].isdecimal()):
        assert template.count('[X]') == 1, template
        prompt = instruction
        for x, y in examples:
            char_span_start = {}
            char_span_end = {}
            
            char_span_start['example'] = len(prompt)
            if template.startswith('[X]'):
                char_span_start['x0'] = len(prompt)
                char_span_end['x0'] = char_span_start['x0'] + len(x)
            else:
                char_span_start['x0'] = len(prompt) + template.index(' [X]')
                char_span_end['x0'] = char_span_start['x0'] + 1 + len(x)
            
            template_with_x = template.replace('[X]', x)
            char_span_start['y'] = len(prompt) + template_with_x.index(' [Y]')
            char_span_end['y'] = char_span_start['y'] + 1 + len(y)
            
            prompt += template_with_x.replace('[Y]', y)
            if len(examples) > 1:
                prompt += '\n'
            else:
                assert len(prompt) == char_span_end['y'], f'len(prompt) ({len(prompt)}) != char_span_end["y"] ({char_span_end["y"]})'
            char_span_end['example'] = len(prompt)
            char_spans.append([char_span_start, char_span_end])
            assert prompt[char_span_start['y']: char_span_end['y']] == f' {y}', f"{prompt[char_span_start['y']: char_span_end['y']]} != ' {y}'"
            if template.startswith('[X]'):
                assert prompt[char_span_start['x0']: char_span_end['x0']] == x, f"{prompt[char_span_start['x0']: char_span_end['x0']]} == '{x}'"
            else:
                assert prompt[char_span_start['x0']: char_span_end['x0']] == f' {x}', f"{prompt[char_span_start['x0']: char_span_end['x0']]} != ' {x}'"
    else:
        raise NotImplementedError(f'rel = {rel}')
    return prompt, char_spans, permutation


def get_results(logprobs, char_spans, all_labels_list, prompt, verbalizer, verbalizer2idx, verbalizer_set, save_verbalizer_logprobs=False):
    results = []
    for (char_start, char_end), all_labels in zip(char_spans, all_labels_list):
        # Get label logprob
        token_start = {}
        token_end = {}
        logprob = {}
        char_start['prey'] = char_start['example']
        char_end['prey'] = char_start['y']
        for data_type in ['y', 'x0', 'x1', 'x2', 'example', 'prey']:
            if data_type not in char_start:
                assert data_type not in char_end
                continue
            assert data_type in char_end

            # Find start token
            if char_start[data_type] in logprobs['text_offset']:
                token_start[data_type] = logprobs['text_offset'].index(char_start[data_type])
            else: # When leading punctuation is tokenized as part of x
                assert (data_type in {'x0', 'x1', 'x2'}) and (prompt[char_start[data_type]] in string.punctuation), f'char_start[data_type] ({char_start[data_type]}) not in logprobs["text_offset"]: {logprobs["text_offset"]}'
                token_start[data_type] = max(offset_no for offset_no, offset in enumerate(logprobs['text_offset']) if offset < char_start[data_type])

            # Find end token
            if char_end[data_type] == len(prompt):
                token_end[data_type] = None
            elif char_end[data_type] in logprobs['text_offset']:
                token_end[data_type] = logprobs['text_offset'].index(char_end[data_type])
            else: # When trailing punctuation is tokenized as part of x
                assert (data_type in {'x0', 'x1', 'x2'}) and (prompt[char_end[data_type] - 1] in string.punctuation), f'char_end[data_type] ({char_end[data_type]}) not in logprobs["text_offset"]: {logprobs["text_offset"]}'
                token_end[data_type] = int(np.argmax(np.array(logprobs['text_offset']) > char_end[data_type]))

            logprob[data_type] = sum(logprobs['token_logprobs'][token_start[data_type]: token_end[data_type]])

        # Get top logprobs for ranking metrics
        top_logprobs = logprobs['top_logprobs'][token_start['y']]
        top_logprob_values = np.array(list(top_logprobs.values()))
        non_top_prob = 1. - np.sum(np.exp(top_logprob_values))
        top_logprobs_keys = {k[1:] for k in top_logprobs.keys() if k[0] == ' '}
        assert len(verbalizer_set) == len(verbalizer)
        missing_verbalizer_tokens = len(verbalizer_set) - len(top_logprobs_keys.intersection(verbalizer_set))
        non_top_verbalizer_token_logprob = -float('inf')
        if missing_verbalizer_tokens > 0:
            non_top_verbalizer_token_logprob = min(np.log(non_top_prob / missing_verbalizer_tokens), min(top_logprob_values))
        verbalizer_logits = np.ones(len(verbalizer)) * non_top_verbalizer_token_logprob
        for k, v in top_logprobs.items():
            if (k[0] == ' ') and (k[1:] in verbalizer2idx):
                verbalizer_logits[verbalizer2idx[k[1:]]] = v
        verbalizer_logprobs = log_softmax(verbalizer_logits)
        try:
            verbalizer_label_index = verbalizer2idx[logprobs['tokens'][token_start['y']][1:]]
        except (ValueError, KeyError):
            verbalizer_label_index = verbalizer2idx[''.join(logprobs['tokens'][token_start['y']: token_end['y']])[1:]]
        verbalizer_pred_index = int(np.argmax(verbalizer_logprobs))
        
        all_logprob_values = np.concatenate([top_logprob_values, np.log((non_top_prob * np.ones(lm_vocab_size - len(top_logprobs)) / (lm_vocab_size - len(top_logprobs))))])
        entropy = -float(np.sum(np.exp(all_logprob_values) * all_logprob_values))
        token2logprob = {token: lprob for token, lprob in top_logprobs.items() if (token[0] == ' ') and (token[1:] in verbalizer_set)}

        # Treat multi-token labels as a single token
        rank_is_upper_bound = False
        label = ''.join(logprobs['tokens'][token_start['y']: token_end['y']])
        if label not in top_logprobs:
            rank_is_upper_bound = logprob['y'] < min(top_logprobs.values())
            top_logprob_values = np.concatenate([top_logprob_values, [logprob['y']]])
            token2logprob[label] = logprob['y']

        all_labels = [f' {l}' for l in all_labels]
        filtered_token2logprob = {token: lprob for token, lprob in token2logprob.items() if (token == label) or (token not in all_labels)}
        if rank_is_upper_bound:
            remaining_prob = 1. - np.exp(list(token2logprob.values())).sum()
            rank = min(len(verbalizer) - 1, len(filtered_token2logprob) - 1 + int(remaining_prob // np.exp(logprob['y'])))
        else:
            # Sort tokens by log prob (removing valid candidates that aren't being tested)
            tokens_sorted = list(sorted(filtered_token2logprob, key=filtered_token2logprob.get, reverse=True))
            rank = tokens_sorted.index(label)

        # Add results
        results.append({
            'nll': -logprob['y'],
            'x0_nll': -logprob['x0'],
            'x1_nll': (-logprob['x1']) if 'x1' in logprob else None,
            'x2_nll': (-logprob['x2']) if 'x2' in logprob else None,
            'example_nll': -logprob['example'],
            'prey': -logprob['prey'],
            'rank': rank,
            'rank_is_upper_bound': rank_is_upper_bound,
            'token2logprob': token2logprob,
            'label': label,
            'all_labels': all_labels,
            'min_logprob': min(top_logprobs.values()),
            'total_logprob': logsumexp(top_logprob_values),
            'entropy': entropy,
            'verbalizer_logprobs': verbalizer_logprobs.tolist() if save_verbalizer_logprobs else None,
            'verbalizer_entropy': -np.sum(verbalizer_logprobs * np.exp(verbalizer_logprobs)),
            'verbalizer_label_index': verbalizer_label_index,
            'verbalizer_pred_index': verbalizer_pred_index,
            'verbalizer_nll': -float(verbalizer_logprobs[verbalizer_label_index]),
            'verbalizer_acc': int(verbalizer_label_index == verbalizer_pred_index),
        })
    result = {f'{k}s': [r[k] for r in results] for k in results[-1].keys()}
    if len(result['nlls']) > 1:
        result['mdl'] = sum(result['nlls'][:-1]) / len(result['nlls'][:-1])
        result['loo'] = result['nlls'][-2]
    result['test'] = result['nlls'][-1]
    return result


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    if cbarlabel is not None:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", size=14)
        for t in cbar.ax.get_yticklabels():
             t.set_fontsize(14)
    else:
        cbar = None

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def is_hamiltonian(p):
    p_rot = [p]
    for _ in range(p.shape[0] - 1):
        p_rot.append(p_rot[-1][p])
    p_rot = np.array(p_rot)
    return len(set(p_rot[:, -1])) == p.shape[0]


def rand_hamiltonian_perm(n, perm_seed=0):
    p_rand = [None] * n
    old_loc = 0
    rem_idxs = list(range(1, n))
    perm_rng = random.Random(perm_seed)
    for idx in range(n - 1):
        new_loc = perm_rng.choice(rem_idxs)
        rem_idxs.pop(rem_idxs.index(new_loc))
        p_rand[new_loc] = old_loc
        old_loc = new_loc
    p_rand[0] = old_loc
    p_rand = np.array(p_rand)
    assert is_hamiltonian(p_rand), f'Expected Hamiltonian Permutation but got {p_rand}'
    return p_rand


def generate_permutations(num_train, num_dev, data_name):
    max_permutations = min(math.factorial(num_train), math.factorial(5))
    if num_dev is not None:
        permutations = list(itertools.permutations(range(num_train)))
        cyclic_permutations = np.random.default_rng(0).permutation(num_dev).tolist()
        return permutations, cyclic_permutations
    
    if (num_train > 5) and (data_name != 'TREx'):
        raise NotImplemented(f'num_train > 5 for {data_name}')

    if num_train <= 5:
        permutations = list(itertools.permutations(range(num_train)))
        ps = list(itertools.permutations(range(num_train)))
        # num_train == 5 option for backward compatibility
        p_rand = np.array(random.Random(0).choice(ps)) if num_train == 5 else np.array(rand_hamiltonian_perm(num_train))
        all_plists = []
        while (len(ps) > 0) and ((len(all_plists) * num_train) < max_permutations):
            plist = [ps.pop(0)]
            for _ in range(num_train - 1):
                plist.append(tuple(plist[-1][i] for i in p_rand))
                ps.pop(ps.index(plist[-1]))
            all_plists.append(plist)
        random.Random(0).shuffle(all_plists)

        all_plists = np.array(all_plists)
        assert np.all(all_plists.sum(2) == all_plists.sum(2)[0, 0]), 'Expected each training set to have one of each sample'
        assert np.all(all_plists.sum(1) == all_plists.sum(1)[0, 0]), 'Expected each position to have one of each sample'
        all_plists = all_plists.reshape(-1, all_plists.shape[-1])
        cyclic_permutations = [permutations.index(tuple(p)) for p in all_plists]
        return permutations, cyclic_permutations
    
    p_rand = np.array(rand_hamiltonian_perm(num_train))
    all_plists = []
    p_rng = random.Random(0)
    for _ in range(int(math.ceil(max_permutations / num_train))):
        perm = list(range(num_train))
        p_rng.shuffle(perm)
        perm = tuple(perm)
        while perm in all_plists:
            print('Resampling...')
            perm = list(range(num_train))
            p_rng.shuffle(perm)
            perm = tuple(perm)
        plist = [perm]
        for _ in range(num_train - 1):
            plist.append(tuple(plist[-1][i] for i in p_rand))
        all_plists += plist
        plist = np.array(plist)
        assert np.all(plist.sum(1) == plist.sum(1)[0]), 'Expected each training set to have one of each sample'
        assert np.all(plist.sum(0) == plist.sum(0)[0]), 'Expected each position to have one of each sample'
    permutations = all_plists
    cyclic_permutations = list(range(max_permutations))
    return permutations, cyclic_permutations


def get_causal_perm_mask(attention_mask):
    batch_size, seq_len = attention_mask.size()

    perm_mask = []
    for b in range(batch_size):
        first_non_pad = min(attention_mask[b].nonzero()).item()
        perm_mask.append(torch.LongTensor([[int((i <= j) or (i < first_non_pad) or (j < first_non_pad)) for j in range(seq_len)] for i in range(seq_len)]).unsqueeze(0))
    return torch.cat(perm_mask, dim=0)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:min(i + n, len(lst))]


def complete_lm(model, tokenizer, prompt, l=0, num_log_probs=100, echo=True):
    """This function runs GPT-2 locally but places the outputs into an json that looks just like the one
     provided by the OpenAI API."""
    xlnet = model.__class__.__name__ == 'XLNetLMHeadModel'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if isinstance(prompt, str):
        prompt = [prompt] # the code below assumes a list
    input_ids = tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True, return_offsets_mapping=True)
    offset_mapping = input_ids.pop('offset_mapping')
    
    # greedily generate l tokens
    if l > 0:
        assert not xlnet, f'Generation not implemented for {model.__class__.__name__}'
        # the generate function can handle left padded inputs automatically in HF
        # total_sequences is now the input + possible generated output
        total_sequences = model.generate(input_ids=input_ids['input_ids'].to(device), attention_mask=input_ids['attention_mask'].to(device), max_length=l + len(input_ids['input_ids'][0]), do_sample=False)
    else:
        assert echo == True and l == 0
        total_sequences = input_ids['input_ids'].to(device)

    # they want the probs of the top tokens
    if num_log_probs is not None:
        # get the logits for the context and the next l tokens
        if xlnet:  # auto left-padding
            perm_mask = get_causal_perm_mask(input_ids['attention_mask'])
            logits = model.forward(input_ids=total_sequences, attention_mask=input_ids['attention_mask'].to(device), perm_mask=perm_mask.to(device), return_dict=True).logits.detach().cpu().float()
        else:
            # we are left padding, so we need to adjust the position IDs for models that aren't usually left-padded
            attention_mask = (total_sequences != tokenizer.pad_token_id).float()
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            logits = model.forward(input_ids=total_sequences, attention_mask=attention_mask, position_ids=position_ids, return_dict=True).logits.detach().cpu().float()
        if not echo:
            # get the top tokens and probs for the generated l tokens
            probs = torch.softmax(logits[:,-l-1:], dim=2).cpu()
        else:
            # get the top tokens and probs for the context and the generated l tokens
            probs = torch.softmax(logits, dim=2).cpu()
        top_probs, top_tokens = torch.topk(probs, k=num_log_probs)
        logprobs = torch.log(probs)
        top_log_probs = torch.log(top_probs)

    # create the return value to resemble OpenAI
    return_json = {}
    choices = []
    if not hasattr(tokenizer, 'id2token'):
        tokenizer.id2token = tokenizer.batch_decode(list(range(tokenizer.vocab_size)))
    for batch_id in range(len(prompt)):
        curr_json = {}
        # text is just the optional context and next l tokens
        if not echo:
            curr_json['text'] = tokenizer.decode(total_sequences[batch_id][-l:], skip_special_tokens=True)
        else:
            curr_json['text'] = tokenizer.decode(total_sequences[batch_id], skip_special_tokens=True)

        # fill the return json with the top tokens and probs to match the OpenAI return value.
        if num_log_probs is not None:
            curr_json['logprobs'] = {}
            curr_json['logprobs']['top_logprobs'] = []
            curr_json['logprobs']['token_logprobs'] = []
            curr_json['logprobs']['tokens'] = []
            
            nonzero_char_end_idxs = offset_mapping[batch_id, :, 1].nonzero()
            start_tok = min(nonzero_char_end_idxs).item() # inclusive
            end_tok = max(nonzero_char_end_idxs).item() + 1 # exclusive
            if not xlnet:
                assert len(offset_mapping[batch_id, :, 1]) == end_tok
            assert torch.all(offset_mapping[batch_id, start_tok: end_tok, 1] != 0).item(), f'Expected nonzero {offset_mapping[batch_id, start_tok: end_tok, 1]}'
            curr_json['logprobs']['text_offset'] = offset_mapping[batch_id, start_tok:end_tok, 0].tolist()
            
            if not echo:
                # cutoff the -1 here because the probs are shifted one over for LMs
                for current_element_top_log_probs, current_element_top_tokens in zip(top_log_probs[batch_id][:-1], top_tokens[batch_id][:-1]):
                    # tokens is a list of the top token at each position
                    curr_json['logprobs']['tokens'].append(tokenizer.decode([current_element_top_tokens[0]]))
                    # token_logprobs is a list of the logprob of the top token at each position
                    curr_json['logprobs']['token_logprobs'].append(current_element_top_log_probs[0].item())
                    # top_logprobs is a list of dicts for the top K tokens. with each entry being {'token_name': log_prob}
                    temp = {}
                    for log_prob, token in zip(current_element_top_log_probs, current_element_top_tokens):
                        token = token.item()
                        log_prob = log_prob.item()
                        token_str = tokenizer.decode(token)
                        temp[token_str] = log_prob
                    curr_json['logprobs']['top_logprobs'].append(temp)
            else:
                # same as not above but small tweaks
                # we add null to the front because for the GPT models, they have null probability for the first token
                # (for some reason they don't have an beginning of sentence token)
                curr_json['logprobs']['top_logprobs'].append(None)
                # cutoff the -1 here because the probs are shifted one over for LMs
                top_tokens_slice = slice(start_tok+int(xlnet), end_tok-1+int(xlnet))
                for current_element_top_log_probs, current_element_top_tokens in zip(top_log_probs[batch_id][top_tokens_slice], top_tokens[batch_id][top_tokens_slice]):
                    current_element_top_tokens_str = [tokenizer.id2token[cur_id] for cur_id in current_element_top_tokens.tolist()]
                    temp = dict(zip(current_element_top_tokens_str, current_element_top_log_probs.tolist()))
                    curr_json['logprobs']['top_logprobs'].append(temp)
                for index in range(start_tok, end_tok):
                    curr_json['logprobs']['tokens'].append(tokenizer.decode([total_sequences[batch_id][index]]))
                curr_json['logprobs']['token_logprobs'].append(None)
                for index in range(start_tok+int(xlnet), end_tok-1+int(xlnet)):
                    log_probs_token_position_j = logprobs[batch_id][index]
                    # probs are left shifted for LMs
                    curr_json['logprobs']['token_logprobs'].append(log_probs_token_position_j[total_sequences[batch_id][index+int(not xlnet)]].item())
                lengths = {k: len(v) for k, v in curr_json['logprobs'].items()}
                assert len(set(lengths.values())) == 1, f'Mismatched lengths: {lengths}'

        choices.append(curr_json)
    return_json['choices'] = choices
    return return_json


def get_model_response(model, tokenizer, prompts, num_log_probs=100, batch_size=4):
    """
    Obtain model's responses on test sentences, given the training examples
    :param prompts: prompts to use
    :param model_name: name of model to use
    :param return_all_prompts: whether to return all the prompts
    :param num_tokens_to_predict_override: whether to override num token to predict
    :param batch_size: number of examples to pass in at once
    :return: a list of dictionaries
    """
    if isinstance(prompts, str):
        prompts = [prompts] # the code below assumes a list
    if model.__class__.__name__ in ['CTRLLMHeadModel']:  # Add control code (could use others)
        prompts = [f'Wikipedia {p.lstrip()}' for p in prompts]
        raise NotImplemented('Ensure that returned text offset values are as expected, given the above addition of a control code.')
    all_raw_answers = []
    for test_chunk_prompts in chunks(prompts, batch_size):
        for choice in complete_lm(model, tokenizer, test_chunk_prompts, 0, num_log_probs=num_log_probs, echo=True)['choices']:
            all_raw_answers.append(choice['logprobs'])
    return all_raw_answers
