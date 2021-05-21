import numpy as np
import os
import json
from time import time
import torch
from tqdm import tqdm

from src.eval.Scorer import Scorer
from src.eval.Writer import Writer

def eval(config, model, batch_iter, scorer):
    '''
    Evaluate model

    :param config:
    :param model:
    :param batch_iter:
    :param scorer:
    :return:
    '''
    model.eval()
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(batch_iter), desc='Eval', disable=os.getenv('USE_SLURM_LOG') == '1'):
            pred_lbl, lbl_logits = model.predict(batch)
            list_idx = batch["input"]["idx"] if isinstance(batch["input"]["idx"], list) else batch["input"]["idx"].cpu().numpy().tolist()
            list_lbl = batch["output"]["true_lbl"] if "true_lbl" in batch["output"] else batch["output"]["lbl"]

            if config.dataset.lower() == 'fewglue/record':
                true_lbl = torch.tensor([1])
                pred_lbl = torch.tensor([list_lbl[0][pred_lbl[0].item()]])
                scorer.add_batch(list_idx, pred_lbl, true_lbl, lbl_logits.cpu().numpy(), None)
            else:
                scorer.add_batch(list_idx, pred_lbl, list_lbl, lbl_logits.cpu().numpy(), None)



def dev_eval(config, model, batcher, num_batches, dict_avg_val=None):
    '''
    Evaluates the accuracy on the dev partition

    :param config:
    :param model:
    :param batcher: batcher to get batches of data
    :param num_batches:
    :param dict_avg_val: dictionary storing metrics

    :return: currrent dev score
    '''

    dict_eval = {}
    dict_eval["num_batches"] = num_batches

    if dict_avg_val is not None:
        dict_eval.update(dict_avg_val)

    # Get train Score
    preds = {}
    if config.eval_train:
        train_scorer = Scorer(config, config.dataset)
        train_iter = batcher.get_eval_train_batch()
        eval(config, model, train_iter, train_scorer)
        _, train_scores = train_scorer.get_score("train")
        dict_eval.update(train_scores)
        train_logits = train_scorer.get_logits()
        preds['train'] = train_scorer.dict_idx2logits_lbl
    else:
        train_logits = None

    # Get dev Score
    if getattr(config, "force_eval_dev", False) or ((config.dataset.lower() != 'fewglue/record') and config.eval_dev and (config.selection_method is None)):
        dev_scorer = Scorer(config, config.dataset)
        dev_iter = batcher.get_dev_batch()
        eval(config, model, dev_iter, dev_scorer)
        score_eval, dev_scores = dev_scorer.get_score("dev")
        dict_eval.update(dev_scores)
        dev_logits = dev_scorer.get_logits()
        preds['dev'] = dev_scorer.dict_idx2logits_lbl
    else:
        print('Skipping dev evaluation!')
        score_eval = 0
        dev_logits = None

    with open(config.dev_score_file, 'a+') as f_out:
        f_out.write(json.dumps(dict_eval))
        f_out.write('\n')

    return score_eval, dev_logits, train_logits, preds

def test_eval(config, model, batcher):
    '''
    Evaluates the accuracy on the test partition

    :param config:
    :param model:
    :param batcher:
    '''

    model.eval()

    dataset_reader = batcher.get_dataset_reader()
    test_writer = Writer(os.path.join(config.exp_dir, "test.json"), dataset_reader)

    with torch.no_grad():
        for idx, batch in enumerate(batcher.get_test_batch()):
            pred_lbl, lbl_logits = model.predict(batch)
            list_idx = batch["input"]["idx"] if isinstance(batch["input"]["idx"], list) else batch["input"][
                "idx"].cpu().numpy().tolist()
            list_lbl = batch["output"]["true_lbl"] if "true_lbl" in batch["output"] else batch["output"]["lbl"]

            if config.dataset.lower() == 'fewglue/record':
                list_idx = batch["input"]["qas_idx"]
                list_lbl = batch["input"]["candidate_entity"]
                test_writer.add_batch(list_idx, pred_lbl, list_lbl, lbl_logits.cpu().numpy())
            else:
                test_writer.add_batch(list_idx, pred_lbl, list_lbl, lbl_logits.cpu().numpy())

    test_writer.flush_file()


def save_labels(batcher):
    dev_file = f'data/{batcher.config.dataset}/dev_labels.json'
    if not os.path.exists(dev_file):
        with open(dev_file, 'w') as f:
            json.dump(get_labels(batcher.get_dev_batch()), f)
        print(f'Saved to: {dev_file}')

    eval_train_file = f'data/{batcher.config.dataset}/eval_train_labels.seed-{batcher.config.train_set_seed}.json'
    if not os.path.exists(eval_train_file):
        with open(eval_train_file, 'w') as f:
            json.dump(get_labels(batcher.get_eval_train_batch()), f)
        print(f'Saved to: {eval_train_file}')


def get_labels(batch_iter):
    idx2lbls = {}
    for batch in batch_iter:
        list_idx = batch["input"]["idx"] if isinstance(batch["input"]["idx"], list) else batch["input"]["idx"].cpu().numpy().tolist()
        assert len(list_idx) == 1, f'Expected len(list_idx) ({len(list_idx)}) == 1'
        list_lbl = batch["output"]["true_lbl"] if "true_lbl" in batch["output"] else batch["output"]["lbl"]
        assert len(list_lbl) == 1, f'Expected len(list_lbl) ({len(list_lbl)}) == 1'
        if list_idx[0] in idx2lbls:
            idx2lbls[list_idx[0]].append(list_lbl[0])
        else:
            idx2lbls[list_idx[0]] = [list_lbl[0]]
    return idx2lbls
