import argparse
import json
import os
import torch
import numpy as np
from transformers import *

from src.data.Batcher import Batcher
from src.adapet import adapet
from src.utils.Config import Config
from src.utils.util import NumpyEncoder
from src.eval.eval_model import dev_eval, save_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', "--exp_dir", required=True)
    parser.add_argument('-i', "--iter", default=None, required=False)
    parser.add_argument("--save_labels", default=False, action='store_true')
    args = parser.parse_args()
    checkpoint = "cur_model.pt" if args.iter is None else f'iter-{args.iter}.model.pt'
    save_prefix = f'eval' if args.iter is None else f'eval.iter-{args.iter}'

    config_file = os.path.join(args.exp_dir, "config.json")
    base_dir, exp_name = args.exp_dir.rsplit('/', 1)
    config = Config(config_file, mkdir=False, kwargs={'base_dir': f'"{base_dir}"', 'exp_name': f'"{exp_name}"'}, update_exp_config=False)
    config.eval_dev = True
    config.eval_train = False
    config.force_eval_dev = True
    dev_pred_dir, dev_pred_file = config.dev_pred_file.rsplit('/', 1)
    if (not args.save_labels) and os.path.exists(f'{dev_pred_dir}/{save_prefix}.{dev_pred_file}'):
        print('Already Done!')
        exit()

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_weight)
    batcher = Batcher(config, tokenizer, config.dataset)
    if args.save_labels:
        save_labels(batcher)
        exit()
    dataset_reader = batcher.get_dataset_reader()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = adapet(config, tokenizer, dataset_reader).to(device)

    model.load_state_dict(torch.load(os.path.join(args.exp_dir, checkpoint)))
    dev_acc, dev_logits, eval_train_logits, preds = dev_eval(config, model, batcher, 0)

    with open(os.path.join(config.exp_dir, f"{save_prefix}.dev_logits.npy"), 'wb') as f:
        np.save(f, dev_logits)
    with open(os.path.join(config.exp_dir, f"{save_prefix}.eval_train_logits.npy"), 'wb') as f:
        np.save(f, eval_train_logits)
    with open(f'{dev_pred_dir}/{save_prefix}.{dev_pred_file}', 'w') as f:
        json.dump(preds, f, cls=NumpyEncoder)
    print('Done!')
