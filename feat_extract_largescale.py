#! /usr/bin/env python3

import torch

import os
import sys

import argparse

from utils import setup_seed
from utils import get_model
from utils import Logger
from utils_ood import make_id_ood
import numpy as np
import torch.nn.functional as F
import time

# ======== fix data type ========
torch.set_default_tensor_type(torch.FloatTensor)

# ======== options ==============
parser = argparse.ArgumentParser(description='Evaluation on clean samples')
# -------- file param. --------------
parser.add_argument('--data_dir',type=str,default='./data/',help='data directory')
parser.add_argument('--logs_dir',type=str,default='./logs/',help='logs directory')
parser.add_argument('--cache_dir',type=str,default='./cache/',help='logs directory')
parser.add_argument('--dataset',type=str,default='CIFAR10',help='data set name')
parser.add_argument('--model_path',type=str,default='./save/',help='saved model path')
parser.add_argument('--supcon',action='store_true',help='extract features from supcon models')
# -------- hyper param. --------
parser.add_argument('--arch',type=str,default='R50',help='model architecture')
parser.add_argument('--seed',type=int,default=0,help='random seeds')
parser.add_argument('--num_classes',type=int,default=10,help='num of classes')
parser.add_argument('--batch_size',type=int,default=256,help='batch size')    
# -------- ood param. --------
parser.add_argument('--score', choices=['MSP', 'ODIN', 'Energy', 'Mahalanobis', 'GradNorm','RankFeat','kNN'], default='MSP')
parser.add_argument('--in_data', choices=['CIFAR10', 'ImageNet'], default='CIFAR10')
parser.add_argument('--in_datadir', type=str, help='in data dir')
parser.add_argument('--out_data', choices=['Texture','iNaturalist','SUN','Places'], default='SVHN')
parser.add_argument('--out_datadir', type=str, help='out data dir')
args = parser.parse_args()

# ======== log writer init. ========
args.dataset = args.in_data
cache_folder = 'supcon' if args.supcon else 'ce'
if not os.path.exists(os.path.join(args.cache_dir,args.dataset,args.arch,cache_folder)):
    os.makedirs(os.path.join(args.cache_dir,args.dataset,args.arch,cache_folder))
args.cache_path = os.path.join(args.cache_dir,args.dataset,args.arch,cache_folder)

setup_seed(args.seed)

testloaderIn, trainloaderIn, out_loader =  make_id_ood(args)

# ======== load network ========
if args.supcon:
    net = get_model(args).cuda()
    checkpoint = torch.load(args.model_path, map_location=torch.device("cpu"))
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['model'].items()}
    net.load_state_dict(state_dict, strict=False)
else:
    net = get_model(args).cuda()
net.eval()
print('-------- MODEL INFORMATION --------')
print('---- arch.: '+args.arch)
print('---- inf. seed.: '+str(args.seed))

batch_size = args.batch_size

FORCE_RUN = False

featdim = {
    'R50': 2048,
    'MNet': 1280,
}[args.arch]


for split, in_loader in [('train', trainloaderIn), ('val', testloaderIn),]:

    cache_name = os.path.join(args.cache_path, f"{split}_in.mmap")
    if FORCE_RUN or not os.path.exists(cache_name):

        feat_log = np.memmap(cache_name, dtype=float, mode='w+', shape=(len(in_loader.dataset), featdim))
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(in_loader):
                inputs = inputs.cuda()
                start_ind = batch_idx * batch_size
                end_ind = min((batch_idx + 1) * batch_size, len(in_loader.dataset))

                score, feature_list = net.feature_list(inputs)
                out = F.adaptive_avg_pool2d(feature_list[-1],(1,1))
                out = torch.flatten(out, 1)

                feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
                if batch_idx % 100 == 0:
                    print(f"Saving features {batch_idx}/{len(in_loader)} at {cache_name}...")

cache_name = os.path.join(args.cache_path, f"{args.out_data}_out.mmap")
if FORCE_RUN or not os.path.exists(cache_name):

    ood_feat_log = np.memmap(cache_name, dtype=float, mode='w+', shape=(len(out_loader.dataset), featdim))
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(out_loader):
            inputs = inputs.cuda()
            start_ind = batch_idx * batch_size
            end_ind = min((batch_idx + 1) * batch_size, len(out_loader.dataset))

            score, feature_list = net.feature_list(inputs)
            out = F.adaptive_avg_pool2d(feature_list[-1],(1,1))
            out = torch.flatten(out, 1)

            ood_feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
            if batch_idx % 100 == 0:
                print(f"Saving features {batch_idx}/{len(out_loader)} at {cache_name}...")

