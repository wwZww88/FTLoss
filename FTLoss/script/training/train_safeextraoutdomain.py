# ExtraOutDomain

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
        "-s", "SafeExtraOutDomain",
        "-sein", str(param["sein"]),
        "-seon", str(param["seon"]),
        "-d", str(gpu_id)
    ]
    
    print(f"🚀 Starting BP={param} on GPU {gpu_id}")
    subprocess.run(cmd)
    print(f"✅ Finished BP={param} on GPU {gpu_id}")

if __name__ == "__main__":
    # params = [{"sein": (20-i)*100, "seon": i*100} for i in range(21) if i!=10]
    params = [
        {'sein': 0, 'seon': 2000},
        {'sein': 100, 'seon': 1900},
    ]

    with Pool(processes=2) as pool:
        pool.map(run_task, params)

    print("🎉 All experiments are completed!")