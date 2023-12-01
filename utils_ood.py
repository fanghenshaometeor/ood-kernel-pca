import torch
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision as tv
import torchvision.transforms as transforms

import numpy as np
import sklearn.metrics as sk

import os


def make_id_ood(args):
    """Returns train and validation datasets."""
    if args.in_data == 'CIFAR10':
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        args.in_datadir = '~/fangkun/data/CIFAR10'
        in_set = tv.datasets.CIFAR10(args.in_datadir, train=False, transform=test_transform, download=True)
        in_set_train = tv.datasets.CIFAR10(args.in_datadir, train=True, transform=test_transform, download=True)

        if args.out_data == 'SVHN':
            args.out_datadir = '~/fangkun/data/ood_data/svhn/'
            out_set = tv.datasets.SVHN(args.out_datadir, split='test', transform=test_transform, download=False)
        elif args.out_data in ['LSUN','iSUN','places365','LSUN_FIX','ImageNet_FIX','ImageNet_resize']:
            args.out_datadir = '~/fangkun/data/ood_data/{}'.format(args.out_data)
            out_set = tv.datasets.ImageFolder(args.out_datadir, transform=test_transform)
        elif args.out_data == 'Texture':
            args.out_datadir = '~/fangkun/data/ood_data/dtd/images'
            out_set = tv.datasets.ImageFolder(args.out_datadir, transform=test_transform)
        elif args.out_data == 'CIFAR100':
            args.out_datadir = '~/fangkun/data/CIFAR100'
            out_set = tv.datasets.CIFAR100(args.out_datadir, train=False, download=True, transform=test_transform)


    elif args.in_data == 'ImageNet':
        args.in_datadir = '~/imagenet/ILSVRC2012_img_val'
        args.in_datadir_train = '~/imagenet/ILSVRC2012_img_train'
        if args.out_data == 'iNaturalist' or args.out_data == 'SUN' or args.out_data == 'Places':
            args.out_datadir = "~/fangkun/data/ood_data/{}".format(args.out_data)
        elif args.out_data == 'Texture':
            args.out_datadir = '~/fangkun/data/ood_data/dtd/images'

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

        in_set = tv.datasets.ImageFolder(args.in_datadir, test_transform)
        out_set = tv.datasets.ImageFolder(args.out_datadir, test_transform)
        in_set_train = tv.datasets.ImageFolder(args.in_datadir_train, test_transform)

    print(f"Using an in-distribution set {args.in_data} with {len(in_set)} images.")
    print(f"Using an out-of-distribution set {args.out_data} with {len(out_set)} images.")

    in_loader = torch.utils.data.DataLoader(
        in_set, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True, drop_last=False)

    in_loader_train = torch.utils.data.DataLoader(
        in_set_train, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True, drop_last=False)

    out_loader = torch.utils.data.DataLoader(
        out_set, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True, drop_last=False)
    
    args.num_classes = len(in_set.classes)

    return in_loader, in_loader_train, out_loader


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level, pos_label=1.):
    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])



def get_measures(in_examples, out_examples):
    num_in = in_examples.shape[0]
    num_out = out_examples.shape[0]

    # logger.info("# in example is: {}".format(num_in))
    # logger.info("# out example is: {}".format(num_out))

    labels = np.zeros(num_in + num_out, dtype=np.int32)
    labels[:num_in] += 1

    examples = np.squeeze(np.vstack((in_examples, out_examples)))
    aupr_in = sk.average_precision_score(labels, examples)
    auroc = sk.roc_auc_score(labels, examples)

    recall_level = 0.95
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    labels_rev = np.zeros(num_in + num_out, dtype=np.int32)
    labels_rev[num_in:] += 1
    examples = np.squeeze(-np.vstack((in_examples, out_examples)))
    aupr_out = sk.average_precision_score(labels_rev, examples)
    return auroc, aupr_in, aupr_out, fpr

# ======== MSP Score
def iterate_data_msp(data_loader, model):
    confs = []
    m = torch.nn.Softmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            logits = model(x)
            conf, _ = torch.max(m(logits), dim=-1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)

# ======== Energy Score
def iterate_data_energy(data_loader, model, temper):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits = model(x)

            conf = temper * torch.logsumexp(logits / temper, dim=1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)

# ======== ReAct score
def iterate_data_react(data_loader, model, threshold, temperature):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            inputs = x.cuda()
            logits = model.forward_threshold(inputs, threshold=threshold)
            conf = temperature * torch.logsumexp(logits / temperature, dim=1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)

# ======== BATS score
def iterate_data_bats(data_loader, model, temper, lam, feature_mean, feature_std, arch):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()            
            # features = model.forward_features(x)
            features = model.feature_list(x)[-1][-1]
            features = torch.flatten(F.adaptive_avg_pool2d(features,(1,1)),1)

            features = torch.where(features<(feature_std*lam+feature_mean),features,feature_std*lam+feature_mean)
            features = torch.where(features>(-feature_std*lam+feature_mean),features,-feature_std*lam+feature_mean)
            
            # logits = model.forward_head(features)
            if arch == 'R50':
                logits = model.fc(features)
            elif arch == 'MNet':
                logits = model.classifier(features)
            
            conf = temper * (torch.logsumexp(logits / temper, dim=1))
            
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)

def val_ood_fuse(net, in_loader, out_loader, error_in, error_out, args):
    
    net.eval()

    if args.fmethod == 'MSP':
        print("Processing in-distribution data...")
        in_scores = iterate_data_msp(in_loader, net)
        print("Processing out-of-distribution data...")
        out_scores = iterate_data_msp(out_loader, net)
    elif args.fmethod == 'Energy':
        print("Processing in-distribution data...")
        in_scores = iterate_data_energy(in_loader, net, args.temperature_energy)
        print("Processing out-of-distribution data...")
        out_scores = iterate_data_energy(out_loader, net, args.temperature_energy)
    elif args.fmethod == 'ReAct':
        print("Processing in-distribution data...")
        in_scores = iterate_data_react(in_loader, net, args.threshold_react, args.temperature_react)
        print("Processing out-of-distribution data...")
        out_scores = iterate_data_react(out_loader, net, args.threshold_react, args.temperature_react)
    elif args.fmethod == 'BATS':
        if args.in_data == 'ImageNet':
            lam = 1.05
            feature_info = np.load(os.path.join(args.logs_dir,args.dataset,args.arch,'fusion',"bats-feat-info.npz"), allow_pickle=True) # .cuda()
            feature_mean = torch.from_numpy(feature_info['feat_mean'].astype(np.float32)).cuda()
            feature_std = torch.from_numpy(feature_info['feat_std'].astype(np.float32)).cuda()
        print("Processing in-distribution data...")
        in_scores = iterate_data_bats(in_loader, net, args.temperature_energy, lam, feature_mean, feature_std, args.arch)
        print("Processing out-of-distribution data...")
        out_scores = iterate_data_bats(out_loader, net, args.temperature_energy, lam, feature_mean, feature_std, args.arch)
    
    in_examples = (1-error_in.reshape(-1,1)) * in_scores.reshape((-1,1))
    out_examples = (1-error_out.reshape(-1,1)) * out_scores.reshape((-1,1))

    auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)

    print('============ Results for {}+{} on {} ============'.format(args.method, args.fmethod, args.out_data))
    print('AUROC: {}'.format(auroc))
    print('AUPR (In): {}'.format(aupr_in))
    print('AUPR (Out): {}'.format(aupr_out))
    print('FPR95: {}'.format(fpr95))

    return fpr95, auroc, aupr_in







