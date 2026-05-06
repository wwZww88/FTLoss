import re
import os
import sys
from functools import partial
from multiprocessing import Pool, current_process

sys.path.append("/home/sxw/FTLoss/src")
from SafetyEval import *
from Utils import *
from Dataset import prepare_dataset


if __name__ == "__main__":

    print_("⏳ Loading test_dataset...")
    _, test_dataset, _, _ = prepare_dataset()
    test_dataset_flat, all_tasks = flatten_test_dataset(test_dataset, merger)
    print_("✅ Dataset ready.")

    device_id = "7"
    tokenizer_Guard, model_Guard = load_llama_guard(device_id)
    print_("✅ Llama_Guard loaded.")

    base_dir = "/home/sxw/FTLoss/"

    strategy_to_eval = [os.path.join(base_dir, strategy) for strategy in os.listdir(base_dir) if 'llama1b-beavertails' in strategy]
    strategy_to_eval.sort()

    for path in strategy_to_eval:

        print_(f"🚀 Starting Eval for {path}")
        generation_result_path = os.path.join(path, "gen", "generation_result_flat.txt")
        safety_result_path = os.path.join(path, "eval_safety")

        try:
            with open(generation_result_path, 'r') as f:
                line = f.read()
            generation_result_flat_list = eval(line)

            safety_results_flat_list = []
            for generation_result_flat in generation_result_flat_list:

                # Remove surrogate
                print_("Removing surrogate...")
                generation_result_flat = [re.sub(r'[\ud800-\udfff]', '', str(t)) if t else "" for t in generation_result_flat]
                print_("Finished removing surrogate...")

                # Start eval safety
                safety_results_flat = batch_eval_safety(generation_result_flat, model_Guard, tokenizer_Guard, batch_size=8)
                safety_results_flat_list.append(safety_results_flat)
            print_(f"✅ Finished Safety Evaluation for {path}")

            save_safety_result(safety_results_flat_list, safety_result_path, test_dataset, all_tasks)
            print_(f"✅ Result saved to {safety_result_path}")

        except Exception as e:
            print_(f"❌ Error in {path}: {e}")

    print_("Clear GPU Cache")
    model_Guard.to("cpu")
    torch.cuda.empty_cache()

    print_("🎉 All evaluations are completed!")