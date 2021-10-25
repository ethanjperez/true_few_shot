# True Few-Shot Learning with Language Models

This codebase supports using language models (LMs) for true few-shot learning: learning to perform a task using a limited number of examples from a single task distribution.
We choose prompts and hyperparameters for few-shot learning methods using no additional held-out data via methods like cross-validation and minimum description length.
The code reproduces the results in our [paper](https://arxiv.org/abs/2105.11447) and supports two forms of few-shot learning:
1. **"In-context" learning** using LMs similar to [GPT-3](https://arxiv.org/abs/2005.14165). Here, we format a few training examples as input to the LM using a natural language "prompt," and we use the LM to predict the next token. We include the code for in-context learning primarily in the top-level directory (largely in `eval_lm.py`). 
2. **Finetuning** via [ADAPET](https://arxiv.org/abs/2103.11955), which learns from supervised examples using a modified classification loss alongside an auxiliary masked LM objective. We include the code for finetuning ADAPET in subdirectories (e.g., `src/` for training/evaluation code).

You can run this codebase with GPT-2/DistilGPT-2 (using HuggingFace Transformers) or GPT-3 (if you have a key from OpenAI). The underlying model you use is abstracted away using a common API. Below, we describe how to reproduce our results, as well as how to download our precomputed results that we used to produce our paper's plots.

## General Setup Instructions
Clone this repo into your current working directory. Then, follow the general setup instructions below:
```bash
cd true_few_shot    # Move into cloned directory
export BASE=$PWD    # Store top-level directory location
mkdir data exp_out  # Make directories for data and experiment results, or symlink to a location that can host large files
```

Continue below to reproduce our prompt selection experiments (with GPT models on LAMA or SuperGLUE). Skip to [Hyperparameter Selection](#true-few-shot-hyperparameter-selection-for-adapet) to reproduce our ADAPET results.

## True Few-Shot Prompt Selection for GPT

First, create a virtual Python 3.7+ environment. We installed and activated a Python 3.7 with Anaconda 3 (downloadable from [docs.anaconda.com](https://docs.anaconda.com/anaconda/install/)) like so:
```bash
conda create -y -n true_few_shot python=3.7
conda activate true_few_shot
# To deactivate the environment, use conda deactivate
```

Next, install the dependencies for this repo:
```
cd $BASE
pip install -r requirements_prompt.txt
```

Then, download data for LAMA experiments (LAMA-UHN data, LAMA and LPAQA prompts, and LAMA vocab):
```bash
cd $BASE/data

# Download LAMA-UHN
wget https://www.cis.uni-muenchen.de/~poerner/blobs/e-bert/LAMA_UHN.zip
unzip LAMA_UHN.zip
mv data/* .
rmdir data
rm LAMA_UHN.zip

# Download LAMA vocab
wget https://dl.fbaipublicfiles.com/LAMA/common_vocab_cased.txt

# Download original LAMA (to get manual prompts)
wget https://dl.fbaipublicfiles.com/LAMA/data.zip
unzip data.zip
mv data/* .
rmdir data
rm data.zip

# Download LPAQA prompts from Google Drive (or download manually from https://drive.google.com/file/d/15ypcAYvQGYRtIQ-GH7qSLNDlJZ2Sqs0H/view?usp=sharing)
gdown --id 15ypcAYvQGYRtIQ-GH7qSLNDlJZ2Sqs0H
unzip lpaqa.zip
rm lpaqa.zip
```

To experiment with GPT2/DistilGPT2 models, you'll then need to install PyTorch to use transformers models (PyTorch download instructions at [pytorch.org](https://pytorch.org/)). (PyTorch installation is not required for GPT3 models.) We used PyTorch 1.7.1 (with CUDA 11.0.194 for GPU inference), installed with the below command:
```
torch_version_suffix="+cu110" # Use "+cu100" for CUDA 10.0, "+cu101" for CUDA 10.1, and "" for CUDA 10.2
pip install torch==1.7.1${torch_version_suffix} torchvision==0.8.2${torch_version_suffix} -f https://download.pytorch.org/whl/torch_stable.html
```

To experiment with GPT3 models, if you have GPT3 access, set your API key as a bash variable (otherwise, you can still run this repo using GPT2 models):
```
export OPENAI_API_KEY="api-key-here"
```

At this point, you'll need to get the results from LM(s) on LAMA or SuperGLUE, in order to later evaluate MDL/CV/Test Accuracy. You can get these results by:
1. Running our true few-shot prompt selection experiments yourself by continuing below, or
2. Loading our pre-run experiments for GPT2 (instead of having to run them yourself), by skipping to [this section](#loading-our-pre-run-experiments-for-prompt-selection)

After either of the above, we'll describe how you can plot the results in our paper.

### Running true few-shot prompt selection experiments yourself

First, move to the top-level directory in this repo (e.g., run `cd $BASE`). From there, the following command will run inference with DistilGPT2 on LAMA-UHN:
```bash
python eval_lm.py --engine distilgpt2 --num_train 5 --seeds 0
```
This command chooses 5 random examples from LAMA-UHN as training examples, randomly orders them in 5!=120 different ways, and appends an unseen (test) example from LAMA-UHN to the end. Then, the code evaluates the log-probability of the correct answer for each train/test example, which we'll use later to compute CV/MDL/Test Accuracy. Below we describe each command line flag:
- `--engine`: specifies the model you'd like to evaluate with
- `--num_train`: specifies the number of training examples you'd like to use (here, 5); using more than 5 will result in evaluating MDL/CV/test accuracy using only a subset of all possible training example permutations, as described in our paper experiments when we use >5 training examples
- `--seeds`: takes a list of integers (e.g., `0 1 2`) and uses each integer as a random seed for sampling a new LAMA training set (per relation); so if you run with `--seeds 0 1 2`, you'll run on LAMA 3 times in total (useful for calculating mean and std. error over several runs).

We use a similar command to run on CB, RTE, and WiC, just adding a couple extra flags:
```bash
python eval_lm.py --engine distilgpt2 --num_train 5 --seeds 0 --data_name super_glue --rels cb rte wic
```
The `--data_name` flag specifies that you want to run on SuperGLUE datasets, and the `--rels` flags specifies which SuperGLUE datasets you'd like to run on. BoolQ is also supported (using `--rels boolq`), but be warned that the inputs are quite long, so which can make running GPT2 models time-consuming and running GPT3 models costly. Other datasets require some extra modification to use their respective few-shot approach described in the GPT3 paper.

At this point, if you've run one of the above `eval_lm.py` commands, you can skip to [Post-processing GPT Results](#post-processing-gpt-results). Below, we show the specific commands we used to reproduce different sets of results in our paper.

For reference, we include a full bash loop you can run to reproduce all of our LAMA experiments that use 5 training examples:
```bash
for ENGINE in 'distilgpt2' 'gpt2' 'gpt2-medium' 'gpt2-large' 'gpt2-xl' 'ada' 'babbage' 'curie' 'davinci'; do  # Different models, in order: DistilGPT2, GPT2 (117M, 345M, 782M, 1.5B), and GPT3 (2.7B, 6.7B, 13B, 175B) 
for NT in 5; do  # 5 training examples
for SEED in 0 1 2 3 4; do  # 5 random seeds to sample different training sets
python eval_lm.py --engine $ENGINE --num_train $NT --seeds $SEED
# Results will be saved to $BASE/data/rel2template2results.data_name-TREx_UHN.engine-$ENGINE.num_train-$NT.sort_by_weight-False/seed-$SEED
done
done
done
```
You'll probably want to parallelize the calls to `eval_lm.py`, since the above will take a while on a single GPU. Running with GPT3 models (`ada`, `babbage`, `curie`, `davinci`) will query the OpenAI API (so you'll be charged for these queries).

Similarly, we include the full bash loop you can run to reproduce our results for true few-shot prompt selection varying the number of training examples used for selection:
```bash
for ENGINE in 'distilgpt2' 'gpt2' 'gpt2-medium' 'gpt2-large' 'gpt2-xl' 'ada' 'babbage'; do  # Here, we don't use curie (13B) and davinci (175B) models for cost reasons 
for NT in 5 10 15 20 30 40; do  # you won't need to run with NT=5 if you ran the above bash loop
for SEED in 0 1 2 3 4; do  # 5 random seeds to sample different training sets
python eval_lm.py --engine $ENGINE --num_train $NT --seeds $SEED
# Results will be saved to $BASE/data/rel2template2results.data_name-TREx_UHN.engine-$ENGINE.num_train-$NT.sort_by_weight-False/seed-$SEED
done
done
done
```

Lastly, we include the bash loop for reproducing our SuperGLUE experiments:
```bash
for ENGINE in 'distilgpt2' 'gpt2' 'gpt2-medium' 'gpt2-large' 'gpt2-xl' 'ada' 'babbage' 'curie' 'davinci'; do 
for NT in 5; do
for SEED in 0 1 2 3 4; do
python eval_lm.py --engine $ENGINE --num_train $NT --seeds $SEED --data_name super_glue --rels cb rte wic 
# Results will be saved to $BASE/data/rel2template2results.data_name-super_glue_UHN.engine-$ENGINE.num_train-$NT.sort_by_weight-False/seed-$SEED
done
done
done
```

### Post-processing GPT Results

We save a lot of statistics about the predictions made by different GPT models using the `eval_lm.py` command above, which would make it time-consuming to load all of the data in every time we'd like to plot different results. Thus, we first extract the stats that we care about (e.g., stats we need to compute cv/mdl) and save them to smaller files like so:
```bash
cd $BASE
DATA_NAME="TREx"     # Use the "TREx" split of LAMA-UHN
FIELDS="nlls ranks"  # names of stats we use to compute cv/mdl/acc ("nlls" for LM Negative Log-Likelihood, "ranks" to get the rank of the true answer, according to the LM -- we will convert this to accuracy later on) 
python extract_fields.py --data_name $DATA_NAME --keys $FIELDS  # see the command line flags in this script, if you'd like to just extract results for a subset of models, training seeds, etc.
```
For SuperGLUE, we extract the stats for computing cv/mdl/acc as follows:
```bash
cd $BASE
DATA_NAME="super_glue"                   # Use super_glue datasets (WiC, RTE, CB)
FIELDS="verbalizer_nlls verbalize_accs"  # names of stats we use to compute cv/mdl/acc ("verbalizer_nlls" to get the NLL of the true answer after eliminating tokens that aren't "verbalizer" tokens or class names; "verbalizer_accs" to get the accuracy when only consider the probabilities of classes with class names instead of all possible tokens) 
python extract_fields.py --data_name $DATA_NAME --keys $FIELDS  # see the command line flags in this script, if you'd like to just extract results for a subset of models, training seeds, etc.
```

Now you can move to [plotting the results from prompt selection](#plotting-results-from-true-few-shot-prompt-selection) using the results from the above commands.

### Loading our pre-run experiments for prompt selection

You can load the results from our GPT-2 evaluation runs, to avoid having to evaluate all of the models yourself:
```bash
cd $BASE/data
gdown --id 1LSvNS_M47a8QcaZ5-ebNadG_ceW4LzIU  # or download manually from https://drive.google.com/file/d/1LSvNS_M47a8QcaZ5-ebNadG_ceW4LzIU/view
tar -xzvf eval_lm_results.tar.gz
mv eval_lm_results/* .
rmdir eval_lm_results
rm eval_lm_results.tar.gz
```
We do not have permission from OpenAI to release our GPT-3 results (please send us an email if this is an issue for you). However, our pre-computed GPT-2 results will still allow you to move on to plotting our results for GPT-2 models below. 

### Plotting results from true few-shot prompt selection

Then, you can compute and save our main paper plots (figures 1-2, as well as 5) for GPT2/DistilGPT2 models like so:
```bash
cd $BASE
python plot_results.py --exp 'TREx-vary_models'
```
The above command will save plots to `$BASE/plots/TREx-vary_models`. If you have also computed GPT-3 results, simply add the "--use_gpt3" flag to plot GPT-3 results as well.

To plot our other results, change the value for the experiment flag `--exp`:
<table>
<tr>
    <td> <b> --exp </b> </td>
    <td> <b> Plots results for... </b> </td>
</tr>
<tr>
    <td> TREx-vary_models </td>
    <td> various model sizes (all sizes; figures 1, 2, 5) </td>
</tr>
<tr>
    <td> TREx-vary_num_train </td>
    <td> various numbers of training examples (all sizes up to 6.7B; figures 3, 4) </td>
</tr>
<tr>
    <td> TREx-vary_criterion </td>
    <td> various model selection criteria (all criteria described in the Appendix) </td>
</tr>
<tr>
    <td> RTE </td>
    <td> RTE </td>
</tr>
<tr>
    <td> CB </td>
    <td> CB </td>
</tr>
<tr>
    <td> WiC </td>
    <td> WiC </td>
</tr>
</table>


## True Few-Shot Hyperparameter Selection for ADAPET

Here, we describe how to choose hyperparameters for [ADAPET](https://arxiv.org/abs/2103.11955), a few-shot learning method that finetunes a language model using a classification loss alongside an auxiliary masked language modeling objective. We use a modified version of their [original code](https://github.com/rrmenon10/ADAPET), which we include in this repo. We now detail how to setup and run our version of the repo to reproduce our results.

ADAPET trains on GPU, so you'll need to ensure that CUDA is installed for NVIDIA GPUs.
First, move to the main directory (`cd $BASE`) and deactivate any existing virtual environments (e.g. `conda deactivate` if using conda). Then run `source bin/init.sh`, which will automatically: 
- Download the [FewGLUE](https://github.com/timoschick/fewglue) and [SuperGLUE](https://super.gluebenchmark.com/tasks) datasets in `data/fewglue/{task}` and `data/superglue/{task}` respectively. 
- Install and setup environment with correct dependencies into a virtual environment.

If you run into issues installing the virtual env with the `source bin/init.sh`, you can also use conda (as we did) to run experiments instead. We installed Python 3.7.10 using Anaconda 4.8.3, as well as the required dependencies (including PyTorch 1.5) like so:
```
conda create -y -n adapet python=3.7.10
conda activate adapet
pip install -r requirements.txt
```
To train with an AMD GPU instead of NVIDIA, you'll need to install ROCm (AMD's CUDA) and then install a ROCm-compatible version of PyTorch 1.8+ instead of PyTorch 1.5 as done above (see [pytorch.org](https://pytorch.org/) for instructions).

We shuffle FewGLUE and generate the 3 additional random subsets of SuperGLUE used in our paper with:
```bash
python subsample_superglue.py
```

For ReCoRD, you'll also need to download separate files containing the few-shot training set labels (we had to format these in a special way for our evaluation scripts later on):
```bash
cd $BASE/data/fewglue/ReCoRD
gdown --id 1oTURVQ0Zvoq5cQr7PtqPMc775bp9o6fz
tar -xzvf eval_train_labels.tar.gz
rm eval_train_labels.tar.gz
mv eval_train_labels/* .
rmdir eval_train_labels
``` 

You can train an ADAPET model on the full WiC dataset like so:
```bash
TN="WiC"     # For other tasks, change to "CB" "COPA" "BoolQ" "RTE" "WSC" "MultiRC" "ReCoRD"
TSS=0        # Random seed for sampling training set. Use TSS=0 for FewGLUE. We used TSS in 0 1 2 3 for our 4 training sets
FMR="False"  # Whether or not to use a fixed masking ratio (as opposed to variable masking ratio -- see ADAPET Appendix Table 12. We always use "False" or variable masking, following ADAPET.)
MA=0.105     # Mask alpha or fraction to use (we sweep over 0.075 0.10 0.105 0.15 following ADAPET)
checkpoint_dirname="tss-$TSS.ma-$MA.fmr-$FMR"
checkpoint_dir="exp_out/fewglue/$TN/albert-xxlarge-v2/$checkpoint_dirname"
mkdir -p $checkpoint_dir  # Make directory for saving results
python src/train.py -c config/$TN.json -k "exp_name='$checkpoint_dirname'" "mask_alpha=$MA" "fixed_mask_ratio=$FMR" "train_set_seed=$TSS" "save_model=True"
```

Evaluate the K=8 fold cross-validation loss for the above dataset/hyperparameters by training 8 models as follows (the code refers to "folds" as "blocks", following terminology in MDL):
```bash
# Set same hyperparameters as before
TN="WiC"
TSS=0
FMR="False"
MA=0.105

SM="cv"      # selection method: "cv" for cross-validation or "mdl" for minimum description length
NB=8         # number of blocks/folds for mdl/cv evaluation 
for BN in $(seq 0 $((NB-1))); do  # iterate over all block (fold) numbers
    checkpoint_dirname="tss-$TSS.ma-$MA.fmr-$FMR.sm-$SM.nb-$NB.bn-$BN"
    checkpoint_dir="exp_out/fewglue/$TN/albert-xxlarge-v2/$checkpoint_dirname"
    mkdir -p $checkpoint_dir
    python src/train.py -c config/$TN.json -k "exp_name='$checkpoint_dirname'" "selection_method='$SM'" "num_blocks=$NB" "block_no=$BN" "mask_alpha=$MA" "fixed_mask_ratio=$FMR" "train_set_seed=$TSS"
done
```

For more details on how to train/evaluate/test ADAPET, please see the [README](https://github.com/rrmenon10/ADAPET#readme) of the original ADAPET repo, which we used for our code (with only minor modifications).

You can load the results from our training runs, to avoid having to train all of the models yourself:
```bash
# Download training run results
cd $BASE/exp_out
rm -r fewglue  # delete any existing results, so we can replace them with our pre-computed results
gdown --id 1Kz5E7v-ejLFeLGSUKPvy9aAPe9_NByZa  # or download manually from https://drive.google.com/file/d/1NAd8nhvpQTl3AYG3jT2ijBtm313JtSDY/view
tar -xzvf fewglue_adapet_results.tar.gz
rm fewglue_adapet_results.tar.gz
```

You can then print the results from CV/MDL hyperparameter selection with:
```bash
cd $BASE
python adapet.py
```
The above command will print a latex table showing the results for the best/worst/mean/median hyperparameters, as well as the CV/MDL-chosen hyperparameters. You can also show a subset of results (e.g., if you haven't trained on all SuperGLUE tasks), by using command line flags:
```bash
cd $BASE
TNS="MultiRC WiC BoolQ"  # Task Names to evaluate on, from {'BoolQ', 'CB', 'COPA', 'RTE', 'WiC', 'WSC', 'MultiRC', 'ReCoRD'} \
TSSS="0 1"  # Train Set Seeds to show mean/std. dev. results for (we used 0 1 2 3 in our paper) \
SMS="cv mdl"  # Selection Methods used to choosen hyperparameters, in {'cv', 'mdl'}
python adapet.py --tns $TNS --tsss $TSSS --sms $SMS
```

Feel free to open an issue if you have any questions, and have fun true few-shot learning!

## Bibtex Citation

```bash
@article{perez2021true,
  author = {Ethan Perez and Douwe Kiela and Kyunghyun Cho},
  title = {True Few-Shot Learning with Language Models},
  journal={NeurIPS},
  year = {2021},
  url = {https://arxiv.org/abs/2105.11447}
}
```
