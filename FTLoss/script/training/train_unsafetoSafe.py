import os

if __name__ == "__main__":
    cmd = [
        "python", "/FTLoss/src/train.py",
        "-s", "BalancedRatio",
        "-brs", str(param["safe"]),
        "-bru", str(param["unsafe"]),
        "-d", str(gpu_id)
    ]

    python -u /FTLoss/src/train.py -s "SafeOnlyRandom" -d "1" -uckp > train_safe2unsafe.log &