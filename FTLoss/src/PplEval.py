import os
import json
from functools import partial

import sys
sys.path.append("/FTLoss/src")

from KnowledgeGraph import *
merger = GraphMerger()

from Dataset import *

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def perplexity(text, tokenizer, model, device):
    """
    Eval model's perplexity for a single input
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad(): 
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss #extracts the loss from the model's outputs
        perplexity = torch.exp(loss) #calculates the perplexity by exponentiating the loss
        
    # print(f"Perplexity: {perplexity.item()}")
        
    return perplexity.item()

def compute_concept_ppl(tokenizer, model, dataset, device):
    """
    Eval model's perplexity for inputs in test_dataset.
    """
    eval_ppl = partial(perplexity, tokenizer=tokenizer, model=model, device=device)
    ppl_result = {}
    
    for concept in dataset.keys():
        # print(f"{concept}, safe-{len(dataset[concept]['safe'])}， unsafe-{len(dataset[concept]['safe'])}")
        
        if not concept in ppl_result.keys():
            ppl_result[concept] = {'safe':[], 'unsafe':[]}
        
        for tag in ['safe', 'unsafe']:
            for id in dataset[concept][tag]:
                # retrieve sample information
                sample = merger.sample_dict[id]
                
                # compute ppl for samples' text
                text = sample['prompt'] + ' ' + sample['response']
                ppl = eval_ppl(text)
                
                ppl_result[concept][tag].append(ppl)
    
    return ppl_result


def eval_model_ppl(model_path, dataset, DEVICE = "4"):
    """
    Evaluate perplexity for model checkpoint series using test_dataset
    !! Not the eval dataset
    Return: ppl_result dict:={
        concept:{
            'safe': [ppl1, ppl2, ppl3, ...],
            'unsafe': [ppl4, ppl5, ...],
            },
    }
    """
    # Specify cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
    device = torch.device(f"cuda:0")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        # pad_token_id=128001,
        local_files_only=True
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        local_files_only=True
        )

    model.to(device)
    print(f"Model running on cuda:{DEVICE}.")

    ppl_result = compute_concept_ppl(tokenizer=tokenizer, model=model, dataset=dataset, device=device)
    
    # End eval, clear model from GPU
    model.to('cpu')
    torch.cuda.empty_cache()
    
    return ppl_result

def save_ppl_result(data, filename):
    """
    Save ppl evaluation results for test dataset
    'pipeline.ipynb'
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to: {filename}")

def load_ppl_result(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    print(f"Loading {filename}")
    return data

def eval_ppl(ckps_path, test_dataset, ppl_result_path, DEVICE = "4"):
    """
    Evaluate through the entire FT process, i.e., for series of checkpoints 
    
    Params:
    ckps_path: /model_ckps/checkpoint-333
    ppl_result_path: ppl_result_list.json
    """
    
    # Sort the checkpoint list by save order
    model_to_eval = [os.path.join(ckps_path, file) for file in os.listdir(ckps_path) if 'checkpoint-' in file]
    model_to_eval = sorted(model_to_eval, key=lambda x: int(x.split('-')[-1]))
    
    ppl_result_list = []
    for model_path in  model_to_eval:
        print(f"Start evaluate {model_path}")
        ppl_result_list.append(
            eval_model_ppl(model_path, test_dataset, DEVICE)
            )
    
    save_ppl_result(ppl_result_list, ppl_result_path)
    # print(f"ppl_result_list saved to {ppl_result_path}")

def eval_ppl_base(model_path, test_dataset, ppl_result_path, DEVICE = "4"):
    """
    Evaluate through the entire FT process, i.e., for series of checkpoints 
    
    Params:
    model: base model oath
    ppl_result_path: ppl_result_list.json
    """
    
    ppl_result_list = []
    print(f"Start evaluate {model_path}")
    ppl_result_list.append(
        eval_model_ppl(model_path, test_dataset, DEVICE)
        )
    
    save_ppl_result(ppl_result_list, ppl_result_path)
    # print(f"ppl_result_list saved to {ppl_result_path}")

if __name__ == "__main__":
    
    _, test_dataset, _, _ = prepare_dataset()

    model_to_eval = [
        "/FTLoss/llama-beavertails-strategy/checkpoint-142",
        "/FTLoss/llama-beavertails-strategy/checkpoint-284",
        "/FTLoss/llama-beavertails-strategy/checkpoint-426"
    ]

    ppl_result_list = {}
    for model_path in  model_to_eval:
        ppl_result_list.append(
            eval_model_ppl(model_path, test_dataset, DEVICE = "4")
            )
        
    save_ppl_result(ppl_result_list, '/FTLoss/llama-beavertails-strategy/ppl_result_list.json')