import os
import sys
import argparse
from tqdm import tqdm

import torch 
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

sys.path.append("/FTLoss/src")
from Prompt import Prompt
from Utils import print_
    
if __name__ == "__main__":
    
    MODEL="EmergentMethods/Phi-3-mini-4k-instruct-graph"
    DATASET="PKU-Alignment/BeaverTails"
    DATASET_=DATASET.split('/')[1].lower()
    SLICE="30k_test"        # 30k_test    30k_train
    
    START=0
    END=3021     #20000     #27185
    TOTAL=END-START
    
    DEVICE=4
    BATCH_SIZE=32

    save_path = f"/FTLoss/results/knowledge/{DATASET_}"
    
    prompt = Prompt()

    ds = load_dataset(DATASET)
    def data():
        for i in range(START, END):
            yield prompt.synthesis(ds[SLICE][i]['prompt'] + ' ' + ds[SLICE][i]['response'])

    torch.random.manual_seed(88) 
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True) 

    pipe = pipeline( 
        "text-generation", 
        model=model, 
        tokenizer=tokenizer,
        device=DEVICE,
    ) 

    generation_args = { 
        "max_new_tokens": 1024, 
        "return_full_text": False, 
        "temperature": 0.0, 
        "do_sample": False, 
    } 
            
    def parse_result(text):
        def m1(t):
            return eval(t)
        def m2(t):
            return eval(t.split('\n\n')[-1])
        def m3(t):
            return eval(t.replace('.\n', '{') + '}')
        def m4(t):
            return eval(t.replace('.\n', '') + '}')
        
        methods = [m1, m2]
        for parser in methods:
            try:
                result = parser(text)
                # print(parser.__name__)
                return result
            except:
                continue
            
        return None
        
    failed_kg = 0
    i = START
    with open(os.path.join(save_path, f"{DATASET_}_{SLICE}_data_knowledge_{START}-{END}.json"), 'w') as f, open(os.path.join(save_path, f"{DATASET_}_{SLICE}_data_knowledge_failed_{START}-{END}.json"), 'w') as f_failed:
        f.write(f"Processing {DATASET_}_{SLICE} knowledge extraction, data {START}-{END}.\n")
        
        if i != 0 and i%100 == 0:
            print_(f"All/ Fail/ Success: {i}/ {failed_kg}({round(failed_kg/i, 4)*100}%)/ {(i-failed_kg)}({round((i-failed_kg)/i, 4)*100}%)")
        
        for output in tqdm(pipe(data(), batch_size=BATCH_SIZE, **generation_args), total=TOTAL):
            data = ds[SLICE][i]
            print_(f"Sample {i}")
            i = i + 1
            
            kg = parse_result(output[0]['generated_text'])
            if kg == None:
                failed_kg += 1
                print_(f'Fail: {data['prompt']} {data['response']}\n')
                # print(f"{output[0]['generated_text']}\n")
                result = {
                    'id': i,
                    'prompt': data['prompt'],
                    'response': data['response'],
                    'category': data['category'],
                    'is_safe': data['is_safe'],
                    'knowledge_graph': kg,
                    'output': output,
                }
                f_failed.write(f'"""{str(result)}"""\n')
            else:
                print_(f'Success: {data['prompt']} {data['response']}\n')
                result = {
                    'id': i,
                    'prompt': data['prompt'],
                    'response': data['response'],
                    'category': data['category'],
                    'is_safe': data['is_safe'],
                    'knowledge_graph': kg,
                }
                f.write(f'"""{str(result)}"""\n')


