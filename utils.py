import copy
import os
import random

import shutil
import sys
import time

import numpy as np
import torch
from torchvision import transforms

from dataset import *
from models import *


__all__ = [
    "setup_model_dataset",
    "save_checkpoint",
    "load_checkpoint",
    "AverageMeter",
    "setup_seed",
    "warmup_lr"
]



def setup_model_dataset(args):
    if args.dataset == "cifar10":
        num_classes = 10
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        train_loader, val_loader, test_loader, retain_train_loader, retain_test_loader, forget_loader = cifar10_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers)
        marked_loader,_, test_loader, _, _, _= cifar10_dataloaders(
            batch_size=args.batch_size, 
            data_dir=args.data,
            num_workers=args.num_workers,
            class_to_replace=args.class_to_replace,
            num_indexes_to_replace=args.num_indexes_to_replace,
            indexes_to_replace=args.indexes_to_replace,
            seed=args.seed,
            only_mark=True,
            shuffle=True,
            no_aug=args.no_aug,
        )

        if args.train_seed is None:
            args.train_seed = args.seed
        setup_seed(args.train_seed)

        model = model_dict[args.arch](num_classes=num_classes)
        
        setup_seed(args.train_seed)
        model.normalize = normalize
        print(model)
        return model, train_loader, val_loader, test_loader, marked_loader
    
    elif args.dataset == "cifar100":
        num_classes = 100
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        train_loader, val_loader, test_loader, retain_train_loader, retain_test_loader, forget_loader = cifar100_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers)
        marked_loader,_, test_loader, _, _, _= cifar100_dataloaders(
            batch_size=args.batch_size, 
            data_dir=args.data,
            num_workers=args.num_workers,
            class_to_replace=args.class_to_replace,
            num_indexes_to_replace=args.num_indexes_to_replace,
            indexes_to_replace=args.indexes_to_replace,
            seed=args.seed,
            only_mark=True,
            shuffle=True,
            no_aug=args.no_aug,
        )

        if args.train_seed is None:
            args.train_seed = args.seed
        setup_seed(args.train_seed)

        model = model_dict[args.arch](num_classes=num_classes)
        
        setup_seed(args.train_seed)
        model.normalize = normalize
        print(model)
        return model, train_loader, val_loader, test_loader, marked_loader

def save_checkpoint(state, save_path,name, is_best, filename='checkpoint.pth.tar'):
    
    file_path = os.path.join(save_path, str(name)+filename)
    
    torch.save(state, file_path)
    if is_best:
        shutil.copyfile(file_path, os.path.join(save_path, str(name)+'model_best.pth.tar'))

def load_checkpoint(device, save_path, name, filename='checkpoint.pth.tar'):
    file_path = os.path.join(save_path, str(name)+filename)
    if os.path.exists(file_path):
        print("=> loading checkpoint '{}'".format(file_path))
        return torch.load(file_path, map_location=device)
    else:
        print("=> no checkpoint found at '{}'".format(file_path))
        return None


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_seed(seed):
    print('Setting up seed: {}'.format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True




def warmup_lr(epoch, step, optimizer, one_epoch_step, args):
    overall_steps = args.warmup * one_epoch_step
    current_steps = epoch * one_epoch_step + step

    lr = args.lr * current_steps / overall_steps
    lr = min(lr, args.lr)

    for p in optimizer.param_groups:
        p["lr"] = lr


def dataset_convert_to_test(dataset, args=None):
    if args.dataset == "TinyImagenet":
        test_transform = transforms.Compose([])
    else:
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    dataset.transform = test_transform
    dataset.train = False