import os
import subprocess
from multiprocessing import Pool, current_process

"""
Run Balanced Strategy for different parameters (here 10->19)
"""

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run_task(bp_value):
    process_id = current_process()._identity[0] - 1 
    available_gpus = ["3", "4"]
    gpu_id = available_gpus[process_id % len(available_gpus)]
    
    cmd = [
        "python", "/FTLoss/src/train.py",
        "--strategy", "Balanced",
        "--balanced_samples_per_concept", str(bp_value),
        "--device", str(gpu_id)
    ]
    
    print(f"🚀 Starting BP={bp_value} on GPU {gpu_id}")
    subprocess.run(cmd)
    print(f"✅ Finished BP={bp_value} on GPU {gpu_id}")

if __name__ == "__main__":
    bp_values = list(range(10, 20)) 

    with Pool(processes=2) as pool:
        pool.map(run_task, bp_values)

    print("🎉 All experiments are completed!")