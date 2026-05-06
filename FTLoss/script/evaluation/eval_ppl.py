import sys
import os
from multiprocessing import Pool, current_process

sys.path.append("/home/sxw/FTLoss/src")
from PplEval import *
from Utils import *
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
    
    
    base_dir = "/home/sxw/FTLoss/"
    # strategy_to_eval = [os.path.join(base_dir, strategy) for strategy in os.listdir(base_dir) if 'llama-beavertails' in strategy]
    strategy_to_eval = [os.path.join(base_dir, strategy) for strategy in os.listdir(base_dir) if 'llama1b-beavertails' in strategy]
    strategy_to_eval.sort()

    # available_gpus = ["2", "3", "4"]
    available_gpus = ["5", "6"]
    
    tasks = []
    for i, path in enumerate(strategy_to_eval):
        gpu = available_gpus[i % len(available_gpus)]
        tasks.append((path, gpu, test_dataset))

    print_(f"Total tasks found: {len(tasks)}. Using GPUs: {available_gpus}")

    with Pool(processes=len(available_gpus)) as pool:
        pool.map(run_single_eval, tasks)

    print_("🎉 All parallel evaluations are completed!")