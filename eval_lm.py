import argparse
from copy import deepcopy
import datasets
from IPython.display import display, HTML
import itertools
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import sys
from tqdm.auto import tqdm, trange
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import rand_hamiltonian_perm, read_jsonl, normalize, preprocess_wsc, get_logprobs, format_prompt, get_results, engine2name, engine2bs


notebook = sys.argv[0].endswith('ipykernel_launcher.py')
if notebook:
    display(HTML("<style>.container { width:100% !important; }</style>"))
DATA_DIR = f'{os.getenv("BASE")}/data'
stat_names = ['mdl', 'loo', 'test']
cm = plt.cm.plasma
gpt3_engines = ['ada', 'babbage', 'curie', 'davinci']
rel2seqlenfactor = {
    'boolq': 30,
    'cb': 30,
    'copa': 30,
    'multirc': 60,
    'record': 60,
    'rte': 30,
    'wic': 30,
    'wsc.fixed': 30,
}


# Settings
parser = argparse.ArgumentParser()
parser.add_argument("--num_train", default=5, type=int, help="The number of training examples to use.")
parser.add_argument("--data_name", default='TREx', choices=['TREx', 'Google_RE', 'super_glue'], type=str, help="The dataset to use.")
parser.add_argument("--engine", default='gpt2', choices=sorted(engine2name), type=str, help="The engine (model) to use.")
parser.add_argument("--seeds", default=[0, 1, 2, 3, 4], nargs='+', type=int, help="The random seeds to use for selecting training data points.")
parser.add_argument("--rels", default=None, nargs='+', type=str, help="The subset of relations (tasks) to use. Use all relations if not supplied.")
parser.add_argument("--shard_no", default=0, type=int, help="The shard number to use (for parallelization).")
parser.add_argument("--num_shards", default=1, type=int, help="The total number of ways to shard the data.")
parser.add_argument("--num_dev", default=None, type=int, help="The number of dev examples to use (set automatically if not provided).")
parser.add_argument("--load_logprobs", default=False, action="store_true", help="Whether or not to load logprobs into memory.")
parser.add_argument("--debug", default=False, action="store_true", help="Whether or not to use debug mode / verbose printing.")
args = parser.parse_args('') if notebook else parser.parse_args()

num_train = args.num_train
data_name = args.data_name
seeds = [0] if args.num_train == 0 else args.seeds
num_seeds = len(seeds)
engine = args.engine
shard_no = args.shard_no
num_shards = args.num_shards
num_dev = args.num_dev
load_logprobs = args.load_logprobs
debug = args.debug


if data_name in ['super_glue']:
    rel2x2validys = None
    rel2fields = {
        'boolq': ['passage', 'question', 'label'],
        'cb': ['premise', 'hypothesis', 'label'],
        'copa': ['choice1', 'choice2', 'premise', 'question', 'label'],
        'multirc': ['paragraph', 'question', 'answer', 'label'],
        'record': ['passage', 'query', 'entities', 'answers'],
        'rte': ['premise', 'hypothesis', 'label'],
        'wic': ['sentence1', 'sentence2', 'word', 'label'],
        'wsc.fixed': ['text_highlighted', 'span2_text', 'label'],
    }
    
    rel2template2prompt_types = {}
    rel2xys = {}
    train_rel2xys = {}
    for rel, fields in rel2fields.items():
        if rel in ['copa', 'multirc', 'record', 'wsc.fixed']:
            continue
        if (args.rels is not None) and (rel not in args.rels):
            continue
        print('Loading', rel)
        data = datasets.load_dataset(data_name, rel)
        if rel == 'wsc.fixed':
            data = preprocess_wsc(data)
        train_rel2xys[rel] = [[d[field] for field in fields] for d in data['train']]
        rel2xys[rel] = [[d[field] for field in fields] for d in data['validation']]
        
        templates = read_jsonl(f'{os.getenv("BASE")}/templates.{data_name}/{rel}.jsonl')
        rel2template2prompt_types[rel] = {}
        for template_no, template_info in enumerate(templates):
            template = template_info.get('instruction', '') + template_info['template']
            if ('verbalizer' in template_info) and (len(template_info['verbalizer']) > 0):
                template += '\n' + '/'.join(template_info['verbalizer'])
            template_info['prompt_types'] = [('gpt3' , 0) if (template_no == 0) else ('pet', template_no - 1)]
            rel2template2prompt_types[rel][template] = template_info
    
elif data_name in ['TREx', 'Google_RE']:
    num_prompts_per_type = 5
    # Load verbalizer
    with open(f'{os.getenv("BASE")}/data/common_vocab_cased.txt') as f:
        vocab_tokens = set(line.strip() for line in f)
    vocab_tokens_sorted = list(sorted(vocab_tokens))

    # Load valid answers to eliminate from candidates
    train_rel2xys = None
    rel2xys = {}
    num_dup = 0
    for filename in os.listdir(f'data/{data_name}_UHN'):
        rel = filename.split('.jsonl')[0]
        data = read_jsonl(f'data/{data_name}_UHN/{filename}')
        rel2xys[rel] = []
        for d in data:
            xy = (d['sub_label'], d['obj_label'])
            if (xy in rel2xys[rel]) and (num_train > 0):
                num_dup += 1
                continue
            rel2xys[rel].append(xy)
    print('# Duplicates Removed:', num_dup)

    # Load x, y pairs
    rel2x2validys = {}
    for filename in os.listdir(f'data/{data_name}'):
        rel = filename.split('.jsonl')[0]
        data = read_jsonl(f'data/{data_name}/{filename}')
        rel2x2validys[rel] = {}
        for d in data:
            x, y = (d['sub_label'], d['obj_label'])
            if x not in rel2x2validys[rel]:
                rel2x2validys[rel][x] = []
            if y not in rel2x2validys[rel][x]:
                rel2x2validys[rel][x].append(y)

    # Load relation info
    relations = read_jsonl(f'data/relations.jsonl')
    rel2info = {relation['relation']: {k: v for k, v in relation.items() if k != 'relation'} for relation in relations}

    # Load all prompts of each different type that end in "[Y]."
    rel2template2prompt_types = {rel: {normalize(info['template']): {'prompt_types': [('manual', 0)]}} if normalize(info['template']).endswith('[Y].') else {}
                            for rel, info in rel2info.items() if rel in rel2xys.keys()}

    for prompt_type in ['mine.paraphrase', 'mine', 'manual.paraphrase']:
        for filename in os.listdir(f'data/lpaqa/{prompt_type}'):
            rel = filename.split('.jsonl')[0]
            with open(f'data/lpaqa/{prompt_type}/{filename}') as f:
                df = pd.DataFrame([json.loads(line.strip()) for line in f])
            templates = []
            for template in df['template']:
                template = normalize(template)
                if template not in templates:
                    templates.append(template)
            templates = [t for t in templates if t.endswith('[Y].')]
            top_templates = templates[:min(len(templates), num_prompts_per_type)]
            for template_no, template in enumerate(top_templates):
                if template not in rel2template2prompt_types[rel]:
                    rel2template2prompt_types[rel][template] = {'prompt_types': []}
                rel2template2prompt_types[rel][template]['prompt_types'].append((prompt_type, template_no))
else:
    raise NotImplementedError(f'data_name = {data_name}')


# Make save directory
if debug:
    num_seeds = 1
assert (num_train > 0) or (num_dev is None), 'Not implemented using num_dev != None for few-shot evaluation'
save_dir = f'{DATA_DIR}/rel2template2results.data_name-{data_name}_UHN.engine-{engine}.num_train-{num_train}.sort_by_weight-False'
if num_dev is not None:
    save_dir += f'.num_dev-{num_dev}'
os.makedirs(save_dir, exist_ok=True)
print('Saving to:', save_dir)

# Generate permutations
max_permutations = min(math.factorial(num_train), math.factorial(5))
if num_dev is not None:
    permutations = list(itertools.permutations(range(num_train)))
    cyclic_permutations = np.random.default_rng(0).permutation(num_dev).tolist()
else:
    if (num_train > 5) and (data_name != 'TREx'):
        raise NotImplemented(f'num_train > 5 for {data_name}')
    
    if num_train <= 5:
        permutations = list(itertools.permutations(range(num_train)))
        ps = list(itertools.permutations(range(num_train)))
        # num_train == 5 option for backward compatibility with earlier experiments
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
    else:
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
print('Permutations:', np.array(all_plists).shape)


# Setup LM if running locally
assert engine in engine2name.keys(), f'Expected engine ({engine}) in {engine2name.keys()}'
tokenizer = AutoTokenizer.from_pretrained('gpt2' if engine in gpt3_engines else engine)
model = None
if engine not in gpt3_engines:
    import torch
    torch.set_grad_enabled(False)  # we don't need gradients since we're just running inference
    model = AutoModelForCausalLM.from_pretrained(engine).eval().to('cuda' if torch.cuda.is_available() else 'cpu')
    if tokenizer.pad_token is None:
        assert tokenizer.eos_token is not None, 'Expected to use EOS token (None) as pad_token'
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    tokenizer.padding_side = 'left'


# Query API
seed2rel2template2results = []
rels = sorted(rel2template2prompt_types)[::-1][shard_no::num_shards]
for seed in seeds:
    rel2template2results = {rel: deepcopy(rel2template2prompt_types[rel]) for rel in rels if len(rel2template2prompt_types[rel]) > 1}
    for rel in tqdm(rels, desc='Relations'):
        # Load cached results if they exist
        os.makedirs(f'{save_dir}/seed-{seed}', exist_ok=True)
        save_filename = f'{save_dir}/seed-{seed}/{rel}.json'
        if (not debug) and os.path.exists(save_filename):
            try:
                with open(save_filename) as f:
                    rel2template2results[rel] = json.load(f)
                if not notebook:
                    rel2template2results[rel].clear()
                    del rel2template2results[rel]
                elif not load_logprobs:  # delete logprobs to save memory
                    for template in rel2template2results[rel]:
                        if 'logprobs' in rel2template2results[rel][template]:
                            rel2template2results[rel][template]['logprobs'].clear()
                            del rel2template2results[rel][template]['logprobs']
                print('Loaded', rel)
                continue
            except json.JSONDecodeError as e:
                print(e)
                print('Removing malformed JSON file:', save_filename)
                os.remove(save_filename)
        
        # Make train/dev set
        permutation_order = list(range(len(permutations)))  # Go through permutations in order by default
        if num_train == 0:
            rng = None
            train_idxs = []
            train = []
            dev = deepcopy(rel2xys[rel][:num_dev])
            dev_idxs = list(range(len(dev)))
        else:
            xys = deepcopy(rel2xys[rel])
            rng = np.random.default_rng(seed)
            if data_name == 'super_glue':
                if num_train != 5:
                    raise NotImplemented('num_train != 5 requires changing change to ensure the same training examples are picked as when num_train == 5')
                permutation_order = cyclic_permutations  # Go through permutations in fixed, random order
                
                # Choose training examples, ensuring at least one of each class available
                train_idxs = rng.permutation(len(train_rel2xys[rel])).tolist()[:num_train]
                train = [deepcopy(train_rel2xys[rel][i]) for i in train_idxs]
                num_classes = len({t[-1] for t in train_rel2xys[rel]})
                while len({t[-1] for t in train}) != num_classes:
                    print(f'Resampling train: Only found {len({t[-1] for t in train})} labels')
                    train_idxs = rng.permutation(len(train_rel2xys[rel])).tolist()[:num_train]
                    train = [deepcopy(train_rel2xys[rel][i]) for i in train_idxs]

                num_evals = max(len(permutations), len(xys))
                dev_idxs = rng.permutation(len(xys)).tolist()
                dev_idxs = [dev_idxs[i % len(dev_idxs)] for i in range(num_evals)]
                dev = [xys[i] for i in dev_idxs]
            else:
                rng.shuffle(xys)
                idxs = rng.permutation(len(xys)).tolist()
                idxs = idxs[5: 5 + len(permutations)] + idxs[:5] + idxs[5 + len(permutations):]
                dev_idxs, train_idxs = idxs[:len(permutations)], idxs[len(permutations): num_train + len(permutations)]
                train = [xys[i] for i in train_idxs]
                dev = [xys[i] for i in dev_idxs][:len(permutations)]

        # Get results
        for template in tqdm(sorted(rel2template2results[rel]), desc=f'{rel} ({len(rel2template2results[rel])} templates)'):
            prompts, char_spanss, train_permutations = [], [], []
            
            # Format prompt and query LM
            pattern = rel2template2results[rel][template].get('template', template)
            instruction = '\n' + rel2template2results[rel][template].get('instruction', '')
            verbalizer = rel2template2results[rel][template].get('verbalizer', []) if data_name == 'super_glue' else vocab_tokens_sorted
            single_token_verbalizer = [tokenizer.decode(tokenizer.encode(v)[0]) for v in verbalizer] if rel == 'wic' else verbalizer
            single_token_verbalizer2idx = {v: i for i, v in enumerate(single_token_verbalizer)}
            single_token_verbalizer_set = set(single_token_verbalizer)
            for didx, d in enumerate(dev):
                prompt, char_spans, train_permutation = format_prompt(train, d, didx if (len(train) == 0) else (didx % len(train)), () if (num_train == 0) else permutations[permutation_order[didx % len(permutation_order)]], verbalizer, rel=rel, instruction=instruction, template=pattern)
                prompts.append(prompt)
                char_spanss.append(char_spans)
                train_permutations.append(train_permutation)
                if debug:
                    print(prompt)
            if debug:
                break
            all_logprobs = get_logprobs(prompts, engine, model=model, tokenizer=tokenizer, batch_size=min(max(1, (engine2bs[engine] // num_train) // rel2seqlenfactor.get(rel, 1)), 120))
            
            for didx, (d, logprobs, prompt, char_spans, train_permutation) in enumerate(zip(dev, all_logprobs, prompts, char_spanss, train_permutations)):
                # Get results
                all_labels_list = [[] if (rel2x2validys is None) else rel2x2validys[rel][xy[0]] for xy in train + [d]]
                result = get_results(logprobs, char_spans, all_labels_list, prompt, single_token_verbalizer, single_token_verbalizer2idx, single_token_verbalizer_set, save_verbalizer_logprobs=(data_name == 'super_glue'))
                
                result['permutation'] = [train_idxs[i] for i in train_permutation] + [dev_idxs[didx]]
                if rel in ['boolq']:
                    result['logprobs'] = None
                    del logprobs
                else:
                    result['logprobs'] = logprobs
                
                # Add to full set of results
                for k, v in result.items():
                    if k not in rel2template2results[rel][template]:
                        rel2template2results[rel][template][k] = []
                    rel2template2results[rel][template][k].append(v)

        # Save results
        if not debug:
            if engine in gpt3_engines:  # Save with logprobs to avoid re-querying the API
                with open(save_filename, 'w') as f:
                    json.dump(rel2template2results[rel], f)
            if not load_logprobs:  # delete logprobs to save memory
                for template in set(rel2template2results[rel].keys()):
                    for k in ['logprobs'] + ([] if data_name == 'super_glue' else ['token2logprob', 'verbalizer_logprobs']):
                        if k in rel2template2results[rel][template]:
                            rel2template2results[rel][template][k].clear()
                            del rel2template2results[rel][template][k]
            if engine not in gpt3_engines:  # Save without logprobs since we an re-evaluate models any time
                with open(save_filename, 'w') as f:
                    json.dump(rel2template2results[rel], f)
            if not notebook:
                for template in set(rel2template2results[rel].keys()):
                    rel2template2results[rel][template].clear()
                    del rel2template2results[rel][template]
                rel2template2results[rel].clear()
                del rel2template2results[rel]
    if notebook:
        seed2rel2template2results.append(rel2template2results)

print('Done!')
