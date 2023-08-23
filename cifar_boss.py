import os
import subprocess
import time
import optuna
import argparse
import torch

def run_trial(gpu_id: int, seed: int, study_name: str, storage_path: str, arch: str, dataset: str):
    command = f"python single_trial.py -a {arch} -d {dataset} --manualSeed={seed} --pretrained --workers=4 --gpu-id={gpu_id} --study-name={study_name} --storage={storage_path}"
    return subprocess.Popen(command, shell=True)

def ensure_study(study_name: str, storage_path: str):
    # Check if the study already exists
    try:
        optuna.load_study(study_name=study_name, storage=storage_path)
        print(f"Study '{study_name}' loaded from existing storage.")
    except KeyError:
        # Create a new study if it doesn't exist
        _ = optuna.create_study(study_name=study_name, storage=storage_path, direction='maximize')
        print(f"Created a new study '{study_name}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run BOSS on Multiple GPUs')
    # Datasets
    parser.add_argument('-d', '--dataset', default='cifar100', type=str,
                        help='Dataset name (e.g., cifar10, cifar100)')

    # Architecture
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16_bn', type=str,
                        help='Model architecture')

    # Miscs
    parser.add_argument('--manualSeed', default=888, type=int, help='manual seed')

    # Optuna-specific arguments
    parser.add_argument('--study-name', default='cifar_study', type=str)
    parser.add_argument('--storage-path', default='sqlite:///optuna_cifar.db', type=str)

    # BOSS-specific arguments
    parser.add_argument('--num-total-trial', default=128, type=int)
    parser.add_argument('--warmup-trials', default=32, type=int)
    parser.add_argument('--pretrained-mode', action='store_true')
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus} [Recommended: 8]")

    ensure_study(args.study_name, args.storage_path)

    processes = [None] * num_gpus

    for active_trials in range(args.num_total_trials):
        if active_trials < args.warmup_trials:
            current_study_name = f'{args.study_name}_warmup'
        else:
            current_study_name = f'{args.study_name}_distillation'

        ensure_study(current_study_name, args.storage_path)

        for i in range(num_gpus):
            if processes[i] is None or processes[i].poll() is not None:
                if active_trials < args.total_trials:
                    processes[i] = run_trial(i, args.manualSeed, args.study_name, args.storage_path, args.arch, args.dataset)
                    active_trials += 1
        time.sleep(1)
