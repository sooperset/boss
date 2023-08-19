import os
import subprocess
import time
import optuna
import argparse

def run_trial(gpu_id: int, seed: int, study_name: str, storage_path: str, arch: str, dataset: str):
    command = f"python single_trial.py -a {arch} -d {dataset} --manualSeed={seed} --pretrained --workers=4 --gpu-id={gpu_id} --study-name={study_name} --storage={storage_path}"
    return subprocess.Popen(command, shell=True)

def ensure_study(study_name: str, storage_path: str):
    # Delete existing database file if it exists
    db_file_path = storage_path.split("///")[-1]
    if os.path.exists(db_file_path):
        os.remove(db_file_path)
        print("Existing storage file deleted.")

    # Create a new study
    study = optuna.create_study(study_name=study_name, storage=storage_path, direction='maximize')
    print("Created a new study.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run BOSS on Multiple GPUs')
    parser.add_argument('--manualSeed', default=888, type=int, help='manual seed')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16_bn', type=str,
                        help='Model architecture')
    parser.add_argument('-d', '--dataset', default='cifar100', type=str,
                        help='Dataset name (e.g., cifar10, cifar100)')
    parser.add_argument('--study-name', default='cifar_study', type=str,
                        help='Optuna study name')
    parser.add_argument('--storage-path', default='sqlite:///optuna_cifar.db', type=str,
                        help='Path to Optuna storage (database)')
    parser.add_argument('--total-trials', default=128, type=int,
                        help='Total number of trials to run')
    args = parser.parse_args()

    active_trials = 0

    ensure_study(args.study_name, args.storage_path)

    processes = [None] * 8

    while active_trials < args.total_trials:
        for i in range(8):
            if processes[i] is None or processes[i].poll() is not None:
                if active_trials < args.total_trials:
                    processes[i] = run_trial(i, args.manualSeed, args.study_name, args.storage_path, args.arch, args.dataset)
                    active_trials += 1
        time.sleep(1)
