import os
import sys

import time
import torch
import numpy as np

import sys
import argparse

from utils import setup_seed
from utils import Logger

import metrics

# ======== options ==============
parser = argparse.ArgumentParser(description='CoP and CoRP for OoD detection')
# -------- file param. --------------
parser.add_argument('--data_dir',type=str,default='./data/',help='data directory')
parser.add_argument('--logs_dir',type=str,default='./logs/',help='logs directory')
parser.add_argument('--cache_dir',type=str,default='./cache/',help='logs directory')
parser.add_argument('--dataset',type=str,default='CIFAR10',help='data set name')
parser.add_argument('--model_path',type=str,default='./save/',help='saved model path')
# -------- hyper param. --------
parser.add_argument('--arch',type=str,default='R18',help='model architecture')
parser.add_argument('--seed',type=int,default=0,help='random seeds')
parser.add_argument('--num_classes',type=int,default=10,help='num of classes')
parser.add_argument('--batch_size',type=int,default=256,help='batch size for training')
parser.add_argument('--supcon',action='store_true',help='the network is trained via supcon')
# -------- ood param. --------
parser.add_argument('--score', choices=['MSP', 'ODIN', 'Energy', 'Mahalanobis', 'GradNorm','RankFeat','kNN'], default='MSP')
parser.add_argument('--in_data', choices=['CIFAR10', 'ImageNet'], default='CIFAR10')
parser.add_argument('--in_datadir', type=str, help='in data dir')
parser.add_argument('--out_data', choices=['SVHN','LSUN','iSUN','Texture','places365'], default='SVHN')
parser.add_argument('--out_datadir', type=str, help='out data dir')
parser.add_argument('--out_datasets',default=['SVHN','LSUN','iSUN','Texture','places365'], nargs="*", type=str)
# -------- CoP, CoRP param. --------
parser.add_argument('--method', choices=['CoP', 'CoRP'], default='CoP')
parser.add_argument('--exp_var_ratio', type=float, help='explained variance ratio to choose q')
parser.add_argument('--gamma', type=float, help='variance of the gaussian kernel in CoRP')
parser.add_argument('--M', type=int, help='mapped dimension of the RFFs in CoRP')
args = parser.parse_args()

args.dataset = args.in_data
cache_folder = 'supcon' if args.supcon else 'ce'
args.cache_path = os.path.join(args.cache_dir,args.dataset,args.arch,cache_folder)
if not os.path.exists(os.path.join(args.logs_dir,args.dataset,args.arch,'eval')):
    os.makedirs(os.path.join(args.logs_dir,args.dataset,args.arch,'eval'))
args.logs_path = os.path.join(args.logs_dir,args.dataset,args.arch,'eval',"{}-{}-ood.log".format(cache_folder,args.method))
sys.stdout = Logger(filename=args.logs_path,stream=sys.stdout)

setup_seed(args.seed)

print()
print("-------- -------- --------")
print("The program starts...")

print()
print("Loading features from {}...".format(args.cache_path))
cache_name = os.path.join(args.cache_path, "train_in.npy")
feat_log = np.load(cache_name, allow_pickle=True)
feat_log = feat_log.T.astype(np.float32)

cache_name = os.path.join(args.cache_path, "val_in.npy")
feat_log_val = np.load(cache_name, allow_pickle=True)
feat_log_val = feat_log_val.T.astype(np.float32)

ood_feat_log_all = {}
for ood_dataset in args.out_datasets:
    cache_name = os.path.join(args.cache_path, f"{ood_dataset}_out.npy")
    ood_feat_log = np.load(cache_name, allow_pickle=True)
    ood_feat_log = ood_feat_log.T.astype(np.float32)
    ood_feat_log_all[ood_dataset] = ood_feat_log
print("Features loaded.")
print("Feature dimension = %d"%feat_log.shape[1])

# -------- such an l2 normalization indicates a feature mapping w.r.t. a cosine kernel
normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))

ftrain = prepos_feat(feat_log)
ftest = prepos_feat(feat_log_val)
food_all = {}
for ood_dataset in args.out_datasets:
    food_all[ood_dataset] = prepos_feat(ood_feat_log_all[ood_dataset])

# -------- one more feature mapping of the RFFs w.r.t. a Gaussian kernel
if args.method == 'CoRP':
    m = ftrain.shape[1]
    gamma, M = args.gamma, args.M
    w = np.sqrt(2*gamma)*np.random.normal(size=(M,m))   # generate M i.i.d. samples from p(w)
    u = 2 * np.pi * np.random.rand(M)
    ftrain = np.sqrt(2/M)*np.cos((ftrain.dot(w.T)+u[np.newaxis,:]))
    ftest = np.sqrt(2/M)*np.cos((ftest.dot(w.T)+u[np.newaxis,:]))
    for ood_dataset, food in food_all.items():
        food_all[ood_dataset] = np.sqrt(2/M)*np.cos((food.dot(w.T)+u[np.newaxis,:]))
    print()
    print("Method: {}".format(args.method))
    print("gamma = %f, M = %d"%(gamma, M))
else:
    print()
    print("Method: {}".format(args.method))

# -------- centralize the mapped features
mu = ftrain.mean(axis=0)
ftrain = ftrain - mu
ftest = ftest - mu
for ood_dataset, food in food_all.items():
    food_all[ood_dataset] = food - mu

# -------- linear PCA
print()
print("Running linear PCA...")
K = ftrain.T.dot(ftrain)
u_full, s, _ = np.linalg.svd(K)
# ---- the reduction dimension q is
# ---- selected according to the explained variance ratio
q, s_accuml = -1, np.zeros(ftrain.shape[1])
for i in range(ftrain.shape[1]):
    s_accuml[i] = sum(s[:i]) / sum(s)
    if i > 0 and q < 0:
        if s_accuml[i-1] < args.exp_var_ratio and s_accuml[i] >= args.exp_var_ratio:
            q = i
print("Linear PCA finished.")
print("explained variance ratio = %f"%args.exp_var_ratio)
print("reduction dimension    q = %d"%q)
print("s_accuml at q-1 = %f"%s_accuml[q-1])
print("s_accuml at q   = %f"%s_accuml[q])
print("s_accuml at q+1 = %f"%s_accuml[q+1])

# -------- reconstruction error for OoD detection
u_q = u_full[:,:q]
reconstruct_in = u_q.dot(u_q.T).dot(ftest.T).T
scores_in = - np.linalg.norm(ftest-reconstruct_in, ord=2, axis=1)

all_results = []
for ood_dataset, food in food_all.items():
    reconstruct_ood = u_q.dot(u_q.T).dot(food.T).T 
    scores_ood = - np.linalg.norm(food-reconstruct_ood, ord=2, axis=1)
    results = metrics.cal_metric(scores_in, scores_ood)
    all_results.append(results)
print()
metrics.print_all_results(all_results, args.out_datasets, '{}'.format(args.method))

print()
print("The program ends...")
print("-------- -------- --------")
print()
