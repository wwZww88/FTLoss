# BalancedRatio

"""
Run Balanced Strategy for different parameters 
(here 10->19)
"""

import os
import subprocess
from multiprocessing import Pool, current_process

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run_task(param):
    process_id = current_process()._identity[0] - 1 
    available_gpus = ["3", "4"]
    gpu_id = available_gpus[process_id % len(available_gpus)]
    
    cmd = [
        "python", "/FTLoss/src/train.py",
        "-s", "BalancedRatio",
        "-brs", str(param["safe"]),
        "-bru", str(param["unsafe"]),
        "-d", str(gpu_id)
    ]
    
    print(f"🚀 Starting BP={param} on GPU {gpu_id}")
    subprocess.run(cmd)
    print(f"✅ Finished BP={param} on GPU {gpu_id}")

if __name__ == "__main__":
    params = [{"safe": 20-i, "unsafe": i} for i in range(21) if i!=10]

    with Pool(processes=2) as pool:
        pool.map(run_task, params)

    print("🎉 All experiments are completed!")