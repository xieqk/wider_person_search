from __future__ import print_function, absolute_import
import os
import sys
import time
import json
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from visdom import Visdom

from reid import data_wider
from reid.dataloader_wider import ImageDataset
from reid import transforms as T
from reid import models
from reid.losses import CrossEntropyLabelSmooth, TripletLoss, DeepSupervision
from reid.utils import AverageMeter, Logger, save_checkpoint, my_pickle
from reid.eval_metrics import evaluate
from reid.samplers import RandomIdentitySampler
from reid.optimizers import init_optim
from utils.timer import Timer


def main(args):
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    assert use_gpu is True, 'GPU not available !'
    print("Currently using GPU {}".format(args.gpu))
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(args.seed)

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_wider.init_img_dataset(root=args.root, name=args.dataset)
    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    pin_memory = True

    valloader = DataLoader(
        ImageDataset(dataset.val, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )
    testloader = DataLoader(
        ImageDataset(dataset.test, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=998)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    resume = './reid/models/trained_models/%s_best_model.pth.tar'%args.arch
    print("Loading checkpoint from '{}'".format(resume))
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'])
    model = nn.DataParallel(model).cuda()

    model.eval()
    _t = Timer()
    with torch.no_grad():
        # val set
        val_feats, val_pids = [], []
        for (imgs, pid, _) in valloader:
            imgs = imgs.cuda()
            _t.tic()
            features = model(imgs)
            _t.toc()
            features = features.data.cpu()
            val_feats.append(features)
            val_pids.extend(pid)
            print('Extract features (validation) ... %d/%d BatchTime: %.3f s, Average: %.3f s(%d imgs)'%(
                len(val_pids), dataset.num_val_imgs, _t.diff, _t.average_time, args.test_batch
            ))
        val_feats = torch.cat(val_feats, 0).numpy()
        val_feat_dict = {}
        for i, pid in enumerate(val_pids):
            val_feat_dict.update({pid:val_feats[i].tolist()})
        my_pickle(val_feat_dict, osp.join('features', 'reid_em_val_%s.pkl'%args.arch))
        # test set
        test_feats, test_pids = [], []
        for (imgs, pid, _) in testloader:
            imgs = imgs.cuda()
            _t.tic()
            features = model(imgs)
            _t.toc()
            features = features.data.cpu()
            test_feats.append(features)
            test_pids.extend(pid)
            print('Extract features (test) ... %d/%d BatchTime: %.3f s, Average: %.3f s(%d imgs)'%(
                len(test_pids), dataset.num_test_imgs, _t.diff, _t.average_time, args.test_batch
            ))
        test_feats = torch.cat(test_feats, 0).numpy()
        print('val feature shape: ', val_feats.shape)
        print('test feature shape: ', test_feats.shape)
        test_feat_dict = {}
        for i, pid in enumerate(test_pids):
            test_feat_dict.update({pid:test_feats[i].tolist()})
        my_pickle(test_feat_dict, osp.join('features', 'reid_em_test_%s.pkl'%args.arch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, default='./data', help="root path to data directory")
    parser.add_argument('-d', '--dataset', type=str, default='wider_exfeat',
                        choices=data_wider.get_names())
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help="number of data loading workers (default: 4)")
    parser.add_argument('--height', type=int, default=256,
                        help="height of an image (default: 256)")
    parser.add_argument('--width', type=int, default=128,
                        help="width of an image (default: 128)")
    parser.add_argument('--test-batch', default=32, type=int, help="test batch size")
    parser.add_argument('-a', '--arch', type=str, default='densenet121', choices=models.get_names())
    parser.add_argument('--seed', type=int, default=1, help="manual seed")
    parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()
    main(args)
