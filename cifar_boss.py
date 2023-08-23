import subprocess
import time
import optuna
import argparse
import torch


def run_trial(gpu_id: int, seed: int, current_study_name: str, storage_path: str, arch: str, dataset: str,
              workers: int, is_warmup_trial: bool, pretrained_mode: bool) -> subprocess.Popen:
    """Run a single trial on a specified GPU with given parameters."""
    pretrained_flag = "--pretrained-mode" if pretrained_mode else ""
    command = (f"python single_trial.py -a {arch} -d {dataset} --manualSeed={seed} --workers={workers} "
               f"--gpu-id={gpu_id} --study-name={current_study_name} --storage={storage_path} "
               f"--is-warmup-trial={is_warmup_trial} {pretrained_flag}")
    return subprocess.Popen(command, shell=True)


def ensure_study(study_name: str, storage_path: str) -> None:
    """Ensure that the study exists in the storage."""
    # Check if the study already exists
    try:
        optuna.load_study(study_name=study_name, storage=storage_path)
        print(f"Study '{study_name}' loaded from existing storage.")
    except KeyError:
        # Create a new study if it doesn't exist
        optuna.create_study(study_name=study_name, storage=storage_path, direction='maximize')
        print(f"Created a new study '{study_name}'.")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run BOSS on Multiple GPUs')

    # Datasets
    parser.add_argument('-d', '--dataset', default='cifar100', type=str, help='Dataset name (e.g., cifar10, cifar100)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    # Architecture
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16_bn', type=str, help='Model architecture')
    # Miscs
    parser.add_argument('--manualSeed', default=888, type=int, help='manual seed')
    # Optuna-specific arguments
    parser.add_argument('--study-name', default='cifar_study', type=str)
    parser.add_argument('--storage-path', default='sqlite:///optuna_cifar.db', type=str)
    # BOSS-specific arguments
    parser.add_argument('--num-total-trial', default=128, type=int)
    parser.add_argument('--num-warmup-trial', default=32, type=int)
    parser.add_argument('--pretrained-mode', action='store_true')

    return parser.parse_args()


def main():
    args = parse_arguments()

    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus} [Recommended: 8]")

    # Ensure that the study exists in the storage
    ensure_study(f'{args.study_name}_warmup', args.storage_path)
    ensure_study(f'{args.study_name}_boss', args.storage_path)

    # Initialize a list to hold the subprocesses for each GPU
    processes = [None] * num_gpus
    trial_num = 0

    # Loop through the total trials
    while trial_num < args.num_total_trial:
        is_warmup_trial = True if trial_num < args.num_warmup_trial else False

        # Define study name based on the trial phase (warmup or boss)
        current_study_name = f'{args.study_name}_warmup' if is_warmup_trial else f'{args.study_name}_boss'

        # Loop through GPUs and launch trials
        for i in range(num_gpus):
            process = processes[i]
            if process is None or process.poll() is not None:
                if trial_num < args.num_total_trial:
                    is_warmup_trial = trial_num < args.num_warmup_trial
                    processes[i] = run_trial(i, args.manualSeed, current_study_name, args.storage_path, args.arch,
                                             args.dataset, args.workers, is_warmup_trial, args.pretrained_mode)
                    trial_num += 1
        time.sleep(1)  # Sleep for a second between trial launches


if __name__ == "__main__":
    main()
