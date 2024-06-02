# 训练受害者模型的代码
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def train_wrn(args):
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['wideresnet', 'resnet'])
    parser.add_argument('--model_root', '-m', type=str)
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'imagenet'])
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch_size', '-bs',type=int, default=1)
    parser.add_argument('--data_f_path', '-f', type=str)
    parser.add_argument('--gradientset_path', type=str, default='./gradients_set/')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()
    
    if args.model == None:
        parser.print_help()
        sys.exit(1)

    if args.model == 'wideresnet':
        args.num_classes = 10
        args.data_f_path = './data/cifar10_seurat_10%/'
        train_wrn(args)
        
        
    elif args.model == 'resnet':
        args.num_classes = 20
        args.data_f_path = './data/subimage_seurat_10%/'
        
    else:
        print('model error')
        