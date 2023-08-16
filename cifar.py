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
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

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
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
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
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
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
parser.add_argument('--no-progress-bar', action='store_true',
                    help='Disable the progress bar')
#Device options
parser.add_argument('--gpu-id', default='0,1,2,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

# Distillation
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--loss-type', type=str, default='l2')
parser.add_argument('--temperature', type=float, default=1)
parser.add_argument('--top-k', default=8, type=int, metavar='N')

# BOSS-specific arguments
parser.add_argument('--num-total-trial', default=10, type=int)
parser.add_argument('--warmup-trials', default=6, type=int)
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

def main(trial_number, gpu_id):
    best_acc = 0
    start_epoch = args.start_epoch

    is_warmup_trial = trial_number < args.warmup_trials
    is_pretrained_mode = args.pretrained_mode

    trial_path = os.path.join(args.checkpoint, f'trial_{trial_number}')
    if not os.path.isdir(trial_path):
        mkdir_p(trial_path)



    # Data
    # print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    multiprocessing_context = get_context('loky')

    data_path = f'./data/gpu{gpu_id}'
    trainset = dataloader(root=data_path, train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, pin_memory=True, persistent_workers=True, multiprocessing_context=multiprocessing_context)

    testset = dataloader(root=data_path, train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=True, multiprocessing_context=multiprocessing_context)

    # Model
    # print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    s_model = deepcopy(model)
    s_model = s_model.to(device)

    t_model = None
    if not is_warmup_trial:
        import random
        completed_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
        top_k_trials = sorted(completed_trials, key=lambda x: x.user_attrs["score"], reverse=True)[:args.top_k]
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

            # args.epochs = args.epochs // 2
            print(f'Reduce epochs by half : {args.epochs}')

            args.alpha = float(args.alpha)
            print(f'Update alpha : {args.alpha}')
        else:
            t_trial_ckpt_path = random.choice(top_k_checkpoint_paths)
            t_state_dict = torch.load(t_trial_ckpt_path)['state_dict']
            t_model.load_state_dict(t_state_dict)
            print(f'Load teacher model from {t_trial_ckpt_path}')
        t_model = t_model.to(device)
    
    # model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    # print('    Total params: %.2fM' % /(sum(p.numel() for p in s_model.parameters())/1000000.0))

    optimizer = optim.SGD(s_model.parameters(), lr=args.lr, momentum=1 - args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.epochs)
    ce_criterion = nn.CrossEntropyLoss()

    if args.loss_type == 'l2':
        consist_criterion = nn.MSELoss()
    elif args.loss_type == 'kl':
        consist_criterion = nn.KLDivLoss(reduction='batchmean')
    else:
        raise NotImplementedError

    logger = Logger(os.path.join(args.checkpoint, f'trial_{trial_number}', 'log.txt'), title=f"{args.dataset}_{args.arch}")
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    # Train and val
    for epoch in range(start_epoch, args.epochs):

        # print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, s_model, t_model, ce_criterion, consist_criterion, optimizer, epoch, use_cuda, device, trial_number)
        test_loss, test_acc = test(testloader, s_model, ce_criterion, epoch, use_cuda, device)

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
            }, is_best, trial_number, checkpoint=args.checkpoint)

        scheduler.step()

    logger.close()

    # print(f'Best acc: {best_acc}')
    best_checkpoint = os.path.join(args.checkpoint, f'trial_{trial_number}', 'model_best.pth.tar')
    return best_acc, best_checkpoint

def train(trainloader, s_model, t_model, ce_criterion, consist_criterion, optimizer, epoch, use_cuda, device, trial_number):
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

    if not args.no_progress_bar:
        bar = Bar('Processing', max=len(trainloader))
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
        
            if args.loss_type == 'l2':
                consist_loss = consist_criterion(s_outputs, t_outputs)
            else:
                consist_loss = consist_criterion(F.log_softmax(s_outputs / args.temperature, dim=1), F.softmax(t_outputs / args.temperature, dim=1)).sum(-1)

            alpha = args.alpha
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

        # plot progress
        if not args.no_progress_bar:
            bar.suffix  = '({batch}/{size}) trial: {trial:3d} | Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | CE Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(trainloader),
                        trial=trial_number,
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=ce_losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
    if not args.no_progress_bar:
        bar.finish()
    return (ce_losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    if not args.no_progress_bar:
        bar = Bar('Processing', max=len(testloader))
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

        # plot progress
        if not args.no_progress_bar:
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(testloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()

    if not args.no_progress_bar:
        bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, trial_number, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    trial_path = os.path.join(checkpoint, f'trial_{trial_number}')
    mkdir_p(trial_path)
    filepath = os.path.join(trial_path, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(trial_path, 'model_best.pth.tar'))

def objective(trial):
    gpu_id = trial.number % torch.cuda.device_count()
    args.gpu_id = str(gpu_id)

    lr = trial.suggest_loguniform('lr', 0.001, 1.0)
    weight_decay = trial.suggest_loguniform('weight_decay', 0.00001, 0.01)
    momentum = trial.suggest_loguniform('momentum', 0.001, 1.0)
    train_batch = trial.suggest_int('batch', 64, 256)
    alpha = trial.suggest_categorical('alpha', [0.25, 0.5, 0.75])

    args.lr = lr
    args.weight_decay = weight_decay
    args.momentum = momentum
    args.train_batch = train_batch
    args.alpha = alpha

    acc, checkpoint_path = main(trial.number, gpu_id)
    trial.set_user_attr("score", acc)
    trial.set_user_attr("checkpoint_path", checkpoint_path)

    return acc

if __name__ == '__main__':
    study.optimize(objective, n_trials=args.num_total_trial, n_jobs=torch.cuda.device_count())

    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial
    print('Value: ', trial.value)
    print('Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')
