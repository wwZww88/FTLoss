import os
import sys
from functools import partial
from multiprocessing import Pool, current_process

sys.path.append("/FTLoss/src")
from SafetyEval import *
from Utils import *
from SFT import *
from Dataset import prepare_dataset


if __name__ == "__main__":
    print_("⏳ Loading test_dataset...")
    _, test_dataset, _, _ = prepare_dataset()
    test_dataset_flat, all_tasks = flatten_test_dataset(test_dataset, merger)
    print_("✅ Dataset ready.")

    generation_base_ = partial(generation_base, test_dataset_flat=test_dataset_flat, all_tasks=all_tasks)

    gpu_id = '4'
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model_path = find_local_model_path(model_name)
    path = "/FTLoss/llama1b-beavertails-BaseModel"
    # path = "/FTLoss/llama3b-beavertails-BaseModel"
    generation_result_path = os.path.join(path, "gen")
    
    
    try:
        generation_base_(
            model_path=model_path, 
            generation_result_path=generation_result_path, 
            device_id=str(gpu_id)
            )
        print_(f"✅ Finished Generation for {path} on GPU {gpu_id}")
    except Exception as e:
        print_(f"❌ Error in {path} on GPU {gpu_id}: {e}")

