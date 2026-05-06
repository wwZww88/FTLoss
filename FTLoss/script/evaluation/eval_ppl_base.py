import sys
import os
from multiprocessing import Pool, current_process

sys.path.append("/FTLoss/src")
from PplEval import *
from Utils import *
from SFT import *
from Dataset import prepare_dataset

def run_single_eval(task_tuple):
    """
    task_tuple: (path, gpu_id, test_dataset)
    """
    path, gpu_id, test_dataset = task_tuple

    print_(f"🚀 Starting Eval for {path} on GPU {gpu_id}")

    eval_ppl_path = os.path.join(path, "eval_ppl")
    ckps_path = os.path.join(path, "model_ckps")
    ppl_result_path = os.path.join(eval_ppl_path, "ppl_result_list.json")

    if not os.path.exists(eval_ppl_path):
        os.makedirs(eval_ppl_path, exist_ok=True)
    
    try:
        eval_ppl(ckps_path, test_dataset, ppl_result_path, DEVICE=str(gpu_id))
        print_(f"✅ Finished Eval for {path} on GPU {gpu_id}")
    except Exception as e:
        print_(f"❌ Error in {path} on GPU {gpu_id}: {e}")

if __name__ == "__main__":

    print_("⏳ Loading test_dataset...")
    _, test_dataset, _, _ = prepare_dataset()
    print_("✅ Dataset ready.")
    
    gpu_id = '3'
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    path = "/FTLoss/llama3b-beavertails-BaseModel"

    """
    gpu_id = '4'
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    path = "/FTLoss/llama1b-beavertails-BaseModel"
    """

    model_path = find_local_model_path(model_name)
    eval_ppl_path = os.path.join(path, "eval_ppl")
    ppl_result_path = os.path.join(eval_ppl_path, "ppl_result_list.json")

    base_dir = "/FTLoss/"
    print_(f"🚀 Starting Eval for {path} on GPU {gpu_id}")

    if not os.path.exists(eval_ppl_path):
        os.makedirs(eval_ppl_path, exist_ok=True)
    
    try:
        eval_ppl_base(model_path, test_dataset, ppl_result_path, DEVICE=str(gpu_id))
        print_(f"✅ Finished Eval for {path} on GPU {gpu_id}")
    except Exception as e:
        print_(f"❌ Error in {path} on GPU {gpu_id}: {e}")