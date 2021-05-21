import os
import random


tns = ['BoolQ', 'CB', 'COPA', 'MultiRC', 'RTE', 'ReCoRD', 'WSC', 'WiC']
data_dir = f'{os.getenv("BASE")}/data'
num_train = 32


# Create other random subsets of FewGLUE
for tn in tns:
    with open(f'{data_dir}/superglue/{tn}/train.jsonl') as f:
        data = [line for line in f]
        
    for seed in [1, 2, 3]:
        idxs = list(range(len(data)))
        random.Random(seed).shuffle(idxs)
        if tn == 'WSC':
            data_subset = []
            for idx in idxs:
                if data[idx].endswith(', "label": true}\n'):
                    data_subset.append(data[idx])
                if len(data_subset) >= num_train:
                    break
        else:
            data_subset = [data[idx] for idx in idxs[:num_train]]
        with open(f'{data_dir}/fewglue/{tn}/train.seed-{seed}.jsonl', 'w') as f:
            f.writelines(''.join(data_subset))

# Shuffle FewGLUE Examples
seed = 0
idxs = list(range(num_train))
random.Random(seed).shuffle(idxs)
for tn in tns:
    with open(f'{data_dir}/fewglue/{tn}/train.jsonl') as f:
        data = [line for line in f]
    assert len(data) == num_train, f'Expected len(data) ({len(data)}) == num_train ({num_train})'
    data_subset = [data[idx] for idx in idxs[:num_train]]
    with open(f'{data_dir}/fewglue/{tn}/train.seed-{seed}.jsonl', 'w') as f:
        f.writelines(''.join(data_subset))
