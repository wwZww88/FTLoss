import os
import subprocess
from multiprocessing import Pool, current_process

"""
Run Llama-1B for different Strategy
"""

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run_task(strategy):
    process_id = current_process()._identity[0] - 1 
    available_gpus = ["6", "7"]
    gpu_id = available_gpus[process_id % len(available_gpus)]
    
    cmd = [
        "python", "/FTLoss/src/train.py",
        "--strategy", str(strategy),
        "--device", str(gpu_id)
    ]
    
    print(f"🚀 Starting BP={strategy} on GPU {gpu_id}")
    subprocess.run(cmd)
    print(f"✅ Finished BP={strategy} on GPU {gpu_id}")

if __name__ == "__main__":
    strategies = ['Random', 'SafeOnlyRandom', 'SafeOnlyBalanced', 'UnsafeOnlyRandom', 'UnsafeOnlyBalanced'] 

    with Pool(processes=2) as pool:
        pool.map(run_task, strategies)

    print("🎉 All experiments are completed!")