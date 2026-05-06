import os
import sys
from functools import partial
from multiprocessing import Pool, current_process

sys.path.append("/FTLoss/src")
from SafetyEval import *
from Utils import *
from Dataset import prepare_dataset


if __name__ == "__main__":
    print_("⏳ Loading test_dataset...")
    _, test_dataset, _, _ = prepare_dataset()
    test_dataset_flat, all_tasks = flatten_test_dataset(test_dataset, merger)
    print_("✅ Dataset ready.")

    generation_ = partial(generation, test_dataset_flat=test_dataset_flat, all_tasks=all_tasks)

    base_dir = "/FTLoss/"

    strategy_to_eval = [os.path.join(base_dir, strategy) for strategy in os.listdir(base_dir) if 'llama-beavertails' in strategy]
    strategy_to_eval.sort()

    for path in strategy_to_eval:
    
        ckps_path = os.path.join(path, "model_ckps")
        generation_result_path = os.path.join(path, "gen")
        
        try:
            generation_(
                ckps_path=ckps_path, 
                generation_result_path=generation_result_path, 
                device_id=str(gpu_id)
                )
            print_(f"✅ Finished Generation for {path} on GPU {gpu_id}")
        except Exception as e:
            print_(f"❌ Error in {path} on GPU {gpu_id}: {e}")

