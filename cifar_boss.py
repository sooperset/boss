from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import optuna
from copy import deepcopy
from joblib.externals.loky.backend.context import get_context

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

study_name = 'cifar_study'
storage_path = 'sqlite:///optuna_cifar.db'

try:
    # Attempt to load an existing study
    study = optuna.load_study(study_name=study_name, storage=storage_path)
    print("Loaded existing study.")
except KeyError:
    # If study does not exist, create a new one
    study = optuna.create_study(study_name=study_name, storage=storage_path, direction='maximize')
    print("Created a new study.")


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar100', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=160, type=int, metavar='N',
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
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=20, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

# Distillation
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--top-k', default=8, type=int, metavar='N')

# BOSS-specific arguments
parser.add_argument('--num-total-trial', default=128, type=int)
parser.add_argument('--warmup-trials', default=32, type=int)
parser.add_argument('--pretrained-mode', action='store_true')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

def main(trial_number, gpu_id, trial_args):
    best_acc = 0
    start_epoch = trial_args.start_epoch

    is_warmup_trial = trial_number < trial_args.warmup_trials
    is_pretrained_mode = trial_args.pretrained_mode

    trial_path = os.path.join(trial_args.checkpoint, f'trial_{trial_number}')
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
    if trial_args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    multiprocessing_context = get_context('loky')

    data_path = f'./data/gpu{gpu_id}'
    trainset = dataloader(root=data_path, train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=trial_args.train_batch, shuffle=True, num_workers=trial_args.workers, pin_memory=True, persistent_workers=True, multiprocessing_context=multiprocessing_context)

    testset = dataloader(root=data_path, train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=trial_args.test_batch, shuffle=False, num_workers=trial_args.workers, pin_memory=True, persistent_workers=True, multiprocessing_context=multiprocessing_context)

    # Model
    model = models.__dict__[trial_args.arch](num_classes=num_classes)

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    s_model = deepcopy(model)
    s_model = s_model.to(device)

    t_model = None
    if not is_warmup_trial:
        import random
        completed_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
        top_k_trials = sorted(completed_trials, key=lambda x: x.user_attrs["score"], reverse=True)[:trial_args.top_k]
        top_k_checkpoint_paths = [trial.user_attrs["checkpoint_path"] for trial in top_k_trials]

        t_model = deepcopy(model)
        if is_pretrained_mode:
            selected_ids = random.sample(range(len(top_k_checkpoint_paths)), 2)
            s_trial_ckpt_path = top_k_checkpoint_paths[selected_ids[0]]
            s_state_dict = torch.load(s_trial_ckpt_path)['state_dict']
            s_model.load_state_dict(s_state_dict)
            print(f'Load student model from {s_trial_ckpt_path}')

            t_trial_ckpt_path = top_k_checkpoint_paths[selected_ids[1]]
            t_state_dict = torch.load(t_trial_ckpt_path)['state_dict']
            t_model.load_state_dict(t_state_dict)
            print(f'Load teacher model from {t_trial_ckpt_path}')

            trial_args.alpha = float(trial_args.alpha)
            print(f'Update alpha : {trial_args.alpha}')
        else:
            t_trial_ckpt_path = random.choice(top_k_checkpoint_paths)
            t_state_dict = torch.load(t_trial_ckpt_path)['state_dict']
            t_model.load_state_dict(t_state_dict)
            print(f'Load teacher model from {t_trial_ckpt_path}')
        t_model = t_model.to(device)

    cudnn.benchmark = True

    optimizer = optim.SGD(s_model.parameters(), lr=trial_args.lr, momentum=1 - trial_args.momentum, weight_decay=trial_args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, trial_args.epochs)
    ce_criterion = nn.CrossEntropyLoss()
    consist_criterion = nn.MSELoss()

    logger = Logger(os.path.join(trial_args.checkpoint, f'trial_{trial_number}', 'log.txt'), title=f"{trial_args.dataset}_{trial_args.arch}")
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    # Train and val
    for epoch in range(start_epoch, trial_args.epochs):
        state['lr'] = scheduler.get_last_lr()[0]

        train_loss, train_acc = train(trainloader, s_model, t_model, ce_criterion, consist_criterion, optimizer, epoch, use_cuda, device, trial_number, trial_args)
        test_loss, test_acc = test(testloader, s_model, ce_criterion, epoch, use_cuda, device, trial_args)

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
            }, is_best, trial_number, checkpoint=trial_args.checkpoint)

        scheduler.step()

    logger.close()

    best_checkpoint = os.path.join(trial_args.checkpoint, f'trial_{trial_number}', 'model_best.pth.tar')
    return best_acc, best_checkpoint

def train(trainloader, s_model, t_model, ce_criterion, consist_criterion, optimizer, epoch, use_cuda, device, trial_number, trial_args):
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

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        s_outputs = s_model(inputs)
        ce_loss = ce_criterion(s_outputs, targets)

        if t_model:
            with torch.no_grad():
                t_outputs = t_model(inputs)
            consist_loss = consist_criterion(s_outputs, t_outputs)

            alpha = trial_args.alpha
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

def test(testloader, model, criterion, epoch, use_cuda, device, trial_args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(testloader):
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

def save_checkpoint(state, is_best, trial_number, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    trial_path = os.path.join(checkpoint, f'trial_{trial_number}')
    mkdir_p(trial_path)
    filepath = os.path.join(trial_path, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(trial_path, 'model_best.pth.tar'))

def generate_trial_args(trial, base_args):
    trial_args = deepcopy(base_args)
    trial_args.lr = trial.suggest_loguniform('lr', 0.001, 1.0)
    trial_args.weight_decay = trial.suggest_loguniform('weight_decay', 0.00001, 0.01)
    trial_args.momentum = trial.suggest_loguniform('momentum', 0.001, 1.0)
    trial_args.train_batch = trial.suggest_int('batch', 64, 256)
    trial_args.alpha = trial.suggest_uniform('alpha', 0.0, 1.0)
    return trial_args

def objective(trial):
    gpu_id = trial.number % torch.cuda.device_count()

    trial_args = generate_trial_args(trial, args)

    acc, checkpoint_path = main(trial.number, gpu_id, trial_args)
    trial.set_user_attr("score", acc)
    trial.set_user_attr("checkpoint_path", checkpoint_path)

    return acc

if __name__ == '__main__':
    study.optimize(objective, n_trials=args.num_total_trial, n_jobs=4)

    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial
    print('Value: ', trial.value)
    print('Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')