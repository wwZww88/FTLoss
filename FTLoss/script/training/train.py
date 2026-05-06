import sys
sys.path.append("/FTLoss/src")

import os
#os.environ["TRANSFORMERS_OFFLINE"] = "1"
#os.environ["HF_DATASETS_OFFLINE"] = "1"

import json
import argparse
from functools import partial

parser = argparse.ArgumentParser()

parser.add_argument('-s', '--strategy', dest='strategy', type=str, default="Random")          # The starting position
parser.add_argument('-d', '--device', dest='device', type=str, default="7")             # The cuda id to run this script

# > Strategy Params

# Random
parser.add_argument('-rn', '--random_n_sample', dest='rn', type=int, default=2000)

# Balanced
parser.add_argument('-bp', '--balanced_samples_per_concept', dest='bp',type=int, default=20)

# BalancedRatio
parser.add_argument('-brs', '--balancedratio_safe_samples_per_concept', dest='brs', type=int, default=10)
parser.add_argument('-bru', '--balancedratio_unsafe_samples_per_concept', dest='bru', type=int, default=10)

# SafeOnlyRandom
parser.add_argument('-srn', '--safeonlyrandom_n_sample', dest='srn', type=int, default=2000)

# SafeOnlyBalanced
parser.add_argument('-sbp', '--safeonlybalanced_samples_per_concept', dest='sbp', type=int, default=20)

# UnsafeOnlyRandom
parser.add_argument('-urn', '--unsafeonlyrandom_n_sample', dest='urn', type=int, default=2000)

# UnsafeOnlyBalanced
parser.add_argument('-ubp', '--unsafeonlybalanced_samples_per_concept', dest='ubp', type=int, default=20)

# ExtraOutDomain
parser.add_argument('-ein', '--extraindomain_n_sample', dest='ein', type=int, default=1000)
parser.add_argument('-eon', '--extraoutdomain_n_sample', dest='eon', type=int, default=1000)

# ExtraOutDomain
parser.add_argument('-sein', '--safeextraindomain_n_sample', dest='sein', type=int, default=1000)
parser.add_argument('-seon', '--safeextraoutdomain_n_sample', dest='seon', type=int, default=1000)

# > Train on Which Model
parser.add_argument('-uckp', '--use_ckps', dest='uckp', action='store_true', default=False)

args = parser.parse_args()

import torch
from datasets import Dataset

# from local
from KnowledgeGraph import *
from Dataset import *
from Utils import *
from SFT import *

merger = GraphMerger()

def create_config(strategy, **overrides):
    """
    Load strategy configuration.
    """
    config = {**dataset_params, 'strategy': strategy}
    if strategy in strategy_params:
        config.update(strategy_params[strategy])
    config.update(overrides)
    return config

def save_folder(strategy, params):
    """
    Folder name for result saving.
    """
    if strategy == 'Random':
        return f"{strategy}_n{params['random_n_sample']}"
    
    elif strategy == 'Balanced':
        return f"{strategy}_perconcept{params['balanced_samples_per_concept']}"
    
    elif strategy == 'BalancedRatio':
        return f"{strategy}_safe{params['balancedratio_safe_samples_per_concept']}_unsafe{params['balancedratio_unsafe_samples_per_concept']}"
    
    elif strategy == 'SafeOnlyRandom':
        return f"{strategy}_n{params['safeonlyrandom_n_sample']}"
    
    elif strategy == 'SafeOnlyBalanced':
        return f"{strategy}_perconcept{params['safeonlybalanced_samples_per_concept']}"
    
    elif strategy == 'UnsafeOnlyRandom':
        return f"{strategy}_n{params['unsafeonlyrandom_n_sample']}"
    
    elif strategy == 'UnsafeOnlyBalanced':
        return f"{strategy}_perconcept{params['unsafeonlybalanced_samples_per_concept']}"
    
    elif strategy == 'ExtraOutDomain':
        return f"{strategy}_in{params['extraindomain_n_sample']}_out{params['extraoutdomain_n_sample']}"
    
    elif strategy == 'SafeExtraOutDomain':
        return f"{strategy}_in{params['safeextraindomain_n_sample']}_out{params['safeextraoutdomain_n_sample']}"
    
    else:
        return strategy
    
def save_sample_ids(ds_train, ds_eval, ds_test, output_dir):
    """
    Record samples that are used in current training.
    """
    train_ids = [sample["id"] for sample in ds_train]
    eval_ids = [sample["id"] for sample in ds_eval]
    test_ids = [sample["id"] for sample in ds_test]
    
    train_is_safe_list = [sample['is_safe'] for sample in ds_train]
    eval_is_safe_list = [sample['is_safe'] for sample in ds_eval]
    test_is_safe_list = [sample['is_safe'] for sample in ds_test]
    
    
    sample_info = {
        'train':{
            "n": len(train_ids),
            "n_safe": train_is_safe_list.count(True),
            "n_unsafe": train_is_safe_list.count(False),
            "ids": train_ids
            },
        'eval': {
            "n": len(eval_ids),
            "n_safe": eval_is_safe_list.count(True),
            "n_unsafe": eval_is_safe_list.count(False),
            "ids": eval_ids
            },
        'test': {
            "n": len(test_ids),
            "n_safe": test_is_safe_list.count(True),
            "n_unsafe": test_is_safe_list.count(False),
            "ids": test_ids
            }
    }
    
    with open(os.path.join(output_dir, "sample_ids.json"), 'w') as f:
        json.dump(sample_info, f, indent=4)


"""
strategy_list = ["Random", "Balanced", "BalancedRatio", "SafeOnlyRandom", "SafeOnlyBalanced", "ExtraOutDomain"]
for strategy in strategy_list:
    ds_train = construct_train_dataset(**create_config(strategy))
    is_safe_list = [sample['is_safe'] for sample in ds_train]
    print(is_safe_list.count(True), is_safe_list.count(False))
"""

def train(strategy, params, device="3", use_local_ckps=False):
    print_(f"SFT under strategy '{strategy}'")
    print_(f"{params}")

    print_(f"Preparing training dataset for {strategy} stategy, together with the eval/test dataset...")
    
    ds_train = construct_train_dataset(**create_config(strategy))
    ds_eval = Dataset.from_list([merger.sample_dict[f"test30k-{i}"] for i in range(1, 1000)])
    ds_test=Dataset.from_list([merger.sample_dict[id] for id in [id for split in test_dataset.values() for id in split['safe']+ split['unsafe']]])

    is_safe_list = [sample['is_safe'] for sample in ds_train]
    print_("Datasets has prepared.")
    print_(f"ds_train: n_safe={is_safe_list.count(True)}, n_unsafe={is_safe_list.count(False)}")

    if use_local_ckps == False:
        output_dir = f"/FTLoss/llama1b-beavertails-{save_folder(strategy, params)}"
    else:
        output_dir = f"/FTLoss/llama-beavertails-UnsafetoSafe-{save_folder(strategy, params)}"
        
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        print("Output dir has exists.")
    print_(f"Results will be saved to {output_dir}")

    save_sample_ids(ds_train, ds_eval, ds_test, output_dir)
    print_(f"Dataset sample info has saved to {os.path.join(output_dir, "sample_ids.json")}")

    # Initialize sft class
    llamasft = Llama_SFT(DEVICE=device, local_ckp=use_local_ckps)

    # print(llamasft.training_args)
    llamasft.training_args.output_dir = os.path.join(output_dir, "model_ckps")
    llamasft.train(train_dataset=ds_train, eval_dataset=ds_eval)
    
    # Clear memory from GPU
    llamasft.model.to('cpu')
    torch.cuda.empty_cache()
    
    
if __name__ == "__main__":
    # Load dataset for training and training_eval
    print_("Import datasets ids...")
    train_dataset, test_dataset, augment_dataset, outdomian_dataset = prepare_dataset()
    print_("Datasets ids has imported.")

    dataset_params = {
        'combine_train_dataset': train_dataset,
        'outdomian_dataset': outdomian_dataset,
    }

    # Parameter setting under different strategies
    strategy_params = {
        'Random': {
            'random_n_sample': args.rn,
        },
        'Balanced': {
            'balanced_samples_per_concept': args.bp,
        },
        'BalancedRatio': {
            'balancedratio_safe_samples_per_concept': args.brs,
            'balancedratio_unsafe_samples_per_concept': args.bru,
        },
        'SafeOnlyRandom': {
            'safeonlyrandom_n_sample': args.srn,
        },
        'SafeOnlyBalanced': {
            'safeonlybalanced_samples_per_concept': args.sbp,
        },
        'UnsafeOnlyRandom': {
            'unsafeonlyrandom_n_sample': args.urn,
        },
        'UnsafeOnlyBalanced': {
            'unsafeonlybalanced_samples_per_concept': args.ubp,
        },
        'ExtraOutDomain': {
            'extraindomain_n_sample': args.ein,
            'extraoutdomain_n_sample': args.eon,
        },
        'SafeExtraOutDomain': {
            'safeextraindomain_n_sample': args.sein,
            'safeextraoutdomain_n_sample': args.seon,
        },
    }
    
    device = args.device
    strategy = args.strategy
    params = strategy_params[strategy]
    use_local_ckps = args.uckp
    
    train(strategy, params, device, use_local_ckps)
