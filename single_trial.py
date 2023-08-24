from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import optuna
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models

from utils import Logger, AverageMeter, accuracy, mkdir_p

import warnings
warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar100', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=128, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120],
                        help='Decrease learning rate at these epochs.') 
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16_bn',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: vgg16_bn)')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

# Distillation
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--top-k', default=8, type=int, metavar='N')

# Optuna-specific arguments
parser.add_argument('--study-name', default='cifar_study', type=str)
parser.add_argument('--storage-path', default='sqlite:///optuna_cifar.db', type=str)

# BOSS-specific arguments
parser.add_argument('--is-warmup-trial', default=False, action='store_true')
parser.add_argument('--pretrained-mode', default=False, action='store_true')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

# Attempt to load an existing optuna study
warmup_study = optuna.load_study(study_name=f'{args.study_name}_warmup', storage=args.storage_path)
boss_study = optuna.load_study(study_name=f'{args.study_name}_boss', storage=args.storage_path)
study = warmup_study if args.is_warmup_trial else boss_study    

def main(trial_number):
    best_acc = 0
    start_epoch = args.start_epoch

    is_warmup_trial = args.is_warmup_trial
    is_pretrained_mode = args.pretrained_mode

    trial_path = os.path.join(args.checkpoint, 'warmup' if is_warmup_trial else 'boss', f'trial_{trial_number}')
    if not os.path.isdir(trial_path):
        mkdir_p(trial_path)



    # Data
    transform_train = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    data_path = f'./data'
    trainset = dataloader(root=data_path, train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root=data_path, train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    if args.arch.startswith('vgg'):
        model = models.__dict__[args.arch](num_classes=num_classes)
    else:
        raise NotImplementedError

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    s_model = deepcopy(model).to(device)
    t_model = None

    # Logic for non-warmup trials
    if not is_warmup_trial:
        import random

        # Get completed trials and sort by score
        completed_trials_from_warmup = warmup_study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
        completed_trials_from_boss = boss_study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
        completed_trials = completed_trials_from_warmup + completed_trials_from_boss
        top_k_trials = sorted(completed_trials, key=lambda x: x.user_attrs["score"], reverse=True)[:args.top_k]
        top_k_checkpoint_paths = [trial.user_attrs["checkpoint_path"] for trial in top_k_trials]

        t_model = deepcopy(model)

        if is_pretrained_mode:
            # Select two random IDs for student and teacher models
            trial_random = random.Random(trial_number)
            selected_ids = trial_random.sample(range(len(top_k_checkpoint_paths)), 2)

            # Load the student model
            s_trial_ckpt_path = top_k_checkpoint_paths[selected_ids[0]]
            s_state_dict = torch.load(s_trial_ckpt_path)['state_dict']
            s_model.load_state_dict(s_state_dict)

            # Load the teacher model
            t_trial_ckpt_path = top_k_checkpoint_paths[selected_ids[1]]
            t_state_dict = torch.load(t_trial_ckpt_path)['state_dict']
            t_model.load_state_dict(t_state_dict)

            # Print loaded paths
            print(f"[BOSS / Trial {trial.number}] Loaded student model ({'/'.join(s_trial_ckpt_path.split('/')[-3:-1])}) and teacher model ({'/'.join(t_trial_ckpt_path.split('/')[-3:-1])}). Alpha: {args.alpha:.2f}")
        else:
            # Load a random teacher model
            trial_random = random.Random(trial_number)
            t_trial_ckpt_path = trial_random.choice(top_k_checkpoint_paths)
            t_state_dict = torch.load(t_trial_ckpt_path)['state_dict']
            t_model.load_state_dict(t_state_dict)

            # Print loaded path
            print(f"[BOSS / Trial {trial.number}] Loaded teacher model ({'/'.join(t_trial_ckpt_path.split('/')[-3:-1])}). Alpha: {args.alpha:.2f}")
        t_model = t_model.to(device)

    cudnn.benchmark = True

    optimizer = optim.SGD(s_model.parameters(), lr=args.lr, momentum=1 - args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.epochs)

    # Create loss criteria
    ce_criterion = nn.CrossEntropyLoss()
    consist_criterion = nn.MSELoss()

    logger = Logger(os.path.join(trial_path, 'log.txt'), title=f"{args.dataset}_{args.arch}")
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        state['lr'] = scheduler.get_last_lr()[0]

        train_loss, train_acc = train(trainloader, s_model, t_model, ce_criterion, consist_criterion, args.alpha, optimizer, is_warmup_trial, use_cuda, device)
        test_loss, test_acc = test(testloader, s_model, ce_criterion, use_cuda, device)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, is_warmup_trial, trial_number, checkpoint=args.checkpoint)

        scheduler.step()

    logger.close()

    best_checkpoint = os.path.join(trial_path, 'model_best.pth.tar')
    return best_acc, best_checkpoint

def train(trainloader, s_model, t_model, ce_criterion, consist_criterion, alpha, optimizer, is_warmup_trial, use_cuda, device):
    # switch to train mode
    s_model.train()

    if t_model:
        t_model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    ce_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    for _, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # Compute student model's output
        s_outputs = s_model(inputs)
        ce_loss = ce_criterion(s_outputs, targets)

        # After warmup trials, compute teacher model's output and calculate consistency loss
        if not is_warmup_trial:
            with torch.no_grad():
                t_outputs = t_model(inputs)
            consist_loss = consist_criterion(s_outputs, t_outputs)

            loss = ce_loss * alpha + (1 - alpha) * consist_loss
        else:
            loss = ce_loss

        # measure accuracy and record loss
        prec1, prec5 = accuracy(s_outputs.data, targets.data, topk=(1, 5))
        ce_losses.update(ce_loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return (ce_losses.avg, top1.avg)

def test(testloader, model, criterion, use_cuda, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for _, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, is_warmup_trial, trial_number, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    trial_path = os.path.join(checkpoint, 'warmup' if is_warmup_trial else 'boss', f'trial_{trial_number}')
    mkdir_p(trial_path)
    filepath = os.path.join(trial_path, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(trial_path, 'model_best.pth.tar'))

if __name__ == '__main__':
    trial = study.ask()
    trial_number = trial.number

    # Suggest hyperparameters
    args.lr = trial.suggest_loguniform('lr', 0.001, 1.0)
    args.weight_decay = trial.suggest_loguniform('weight_decay', 0.00001, 0.01)
    args.momentum = trial.suggest_loguniform('momentum', 0.001, 1.0)
    args.train_batch = trial.suggest_int('batch', 64, 256)

    # Suggest alpha hyperparameter if it's not a warmup trial
    if not args.is_warmup_trial:
        args.alpha = trial.suggest_float('alpha', 0.0, 1.0)

    # Run the main trial function
    acc, checkpoint_path = main(trial_number)

    # Record trial results
    trial.set_user_attr("score", acc)
    trial.set_user_attr("checkpoint_path", checkpoint_path)
    study.tell(trial, acc)

    # Retrieve the best trial
    best_trial = study.best_trial

    # Construct and print result summary
    text = f"[WarmUp / Trial {trial.number}] " if args.is_warmup_trial else f"[BOSS / Trial {trial.number}] "
    text += f"LR={args.lr:.6f}, Weight Decay={args.weight_decay:.4f}, "
    text += f"Momentum={args.momentum:.4f}, Batch={args.train_batch}"

    if not args.is_warmup_trial:
        text += f", Alpha={args.alpha:.2f}"

    text += f" ACC: {acc:.4f} | Best - [{'WarmUp' if args.is_warmup_trial else 'BOSS'} / Trial {best_trial.number}] ACC: {best_trial.value:.4f}"
    

    print(text)