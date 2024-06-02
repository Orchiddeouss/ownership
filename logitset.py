import argparse
import torch
import numpy as np
import random
import torch.optim as optim
import time
from splitdata import feature_data
from modelloader import get_model
from getlogits import get_logits
import os
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['wrn28-10', 'resnet34-imgnet'])
    parser.add_argument('--model_root', '-m', type=str)
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'imagenet'])
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch_size', '-bs',type=int, default=1)
    parser.add_argument('--data_f_path', '-f', type=str)
    parser.add_argument('--logits_path', type=str, default='./logits_set/')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()
    
    if args.model == None:
        parser.print_help()
        sys.exit(1)

    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.data_f_path = './data/cifar10_seurat_10%/'
    elif args.dataset == 'imagenet':
        args.num_classes = 20
        args.data_f_path = './data/subimage_seurat_10%/'
    else:
        raise('no such dataset')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    start_time = time.time()
    if args.gpu != -1:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu != -1 else 'cpu'
    
    print(args)

    print('load model')
    model = get_model(args)
    model.to(device)

    if device == 'cpu':
        model.load_state_dict(torch.load(args.model_root, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(args.model_root))
    


    print('get logits from model')
    # get feature embedded images
    train_f_loader = feature_data(args)

    # get logits from model
    g_f = get_logits(model, train_f_loader, device)
    g_f = np.concatenate(g_f, axis=0)
    print(g_f)

    print('save logits')
    model_name = args.model_root.split('/')[3]
    f_data_name = args.data_f_path.split('/')[2]
    g_f_path = args.logits_path + model_name + '/' + f_data_name

    if not os.path.exists(g_f_path):
        os.makedirs(g_f_path)

    np.save(g_f_path+'/g_f.npy', g_f)

    print('Time cost: {} sec'.format(round(time.time() - start_time, 2)))
