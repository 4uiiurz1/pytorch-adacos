import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import joblib
from collections import OrderedDict
from glob import glob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils import *
from omniglot import archs, dataset
import metrics


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', default='ResNet_IR',
                        choices=archs.__all__,
                        help='model architecture')
    parser.add_argument('--backbone', default='resnet18')
    parser.add_argument('--metric', default='adacos',
                        choices=['adacos', 'arcface', 'sphereface', 'cosface', 'softmax'])
    parser.add_argument('--num-features', default=512, type=int,
                        help='dimention of embedded features')
    parser.add_argument('--num-classes', default=1623, type=int)
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float)
    parser.add_argument('--min-lr', default=1e-3, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--nesterov', default=False, type=str2bool)
    parser.add_argument('--cpu', default=False, type=str2bool)

    args = parser.parse_args()

    return args


def train(args, train_loader, model, metric_fc, criterion, optimizer):
    losses = AverageMeter()
    acc1s = AverageMeter()
    acc5s = AverageMeter()

    model.train()
    metric_fc.train()

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        if args.cpu:
            input = input.cpu()
            target = target.long().cpu()
        else:
            input = input.cuda()
            target = target.long().cuda() 

        feature = model(input)
        if args.metric == 'softmax':
            output = metric_fc(feature)
        else:
            output = metric_fc(feature, target)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        acc1s.update(acc1.item(), input.size(0))
        acc5s.update(acc5.item(), input.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('acc@1', acc1s.avg),
        ('acc@5', acc5s.avg),
    ])

    return log


def validate(args, val_loader, model, metric_fc, criterion):
    losses = AverageMeter()
    acc1s = AverageMeter()
    acc5s = AverageMeter()

    # switch to evaluate mode
    model.eval()
    metric_fc.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            if args.cpu:
                input = input.cpu()
                target = target.long().cpu()
            else:
                input = input.cuda()
                target = target.long().cuda() 

            feature = model(input)
            output = metric_fc(feature)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            acc1s.update(acc1.item(), input.size(0))
            acc5s.update(acc5.item(), input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('acc@1', acc1s.avg),
        ('acc@5', acc5s.avg),
    ])

    return log


def main():
    args = parse_args()

    if args.name is None:
        args.name = 'omniglot_%s_%s_%dd' % (
            args.arch, args.metric, args.num_features)

    if not os.path.exists('models/%s' % args.name):
        os.makedirs('models/%s' % args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' % args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' % (arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' % args.name)

    if args.cpu:
        criterion = nn.CrossEntropyLoss().cpu()
    else:
        criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True

    img_paths = glob('omniglot/omniglot/python/images_background/*/*/*.png')
    img_paths.extend(
        glob('omniglot/omniglot/python/images_evaluation/*/*/*.png'))
    labels = LabelEncoder().fit_transform(
        [p.split('/')[-3] + '_' + p.split('/')[-2] for p in img_paths])
    print(len(np.unique(labels)))

    train_img_paths, test_img_paths, train_labels, test_labels = train_test_split(
        img_paths, labels, test_size=0.2, random_state=41, stratify=labels)

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(114),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize(114),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_set = dataset.Omniglot(
        train_img_paths,
        train_labels,
        transform=transform_train)

    test_set = dataset.Omniglot(
        test_img_paths,
        test_labels,
        transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8)

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8)

    # create model
    model = archs.__dict__[args.arch](args)
    if args.cpu:
        model = model.cpu()
    else:
        model = model.cuda()

    if args.metric == 'adacos':
        metric_fc = metrics.AdaCos(
            num_features=args.num_features, num_classes=args.num_classes)
    elif args.metric == 'arcface':
        metric_fc = metrics.ArcFace(
            num_features=args.num_features, num_classes=args.num_classes)
    elif args.metric == 'sphereface':
        metric_fc = metrics.SphereFace(
            num_features=args.num_features, num_classes=args.num_classes)
    elif args.metric == 'cosface':
        metric_fc = metrics.CosFace(
            num_features=args.num_features, num_classes=args.num_classes)
    else:
        metric_fc = nn.Linear(args.num_features, args.num_classes)
    if args.cpu:
        metric_fc = metric_fc.cpu()
    else:
        metric_fc = metric_fc.cuda()

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                               T_max=args.epochs, eta_min=args.min_lr)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'acc@1', 'acc@5', 'val_loss', 'val_acc1', 'val_acc5'
    ])

    best_loss = float('inf')
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch + 1, args.epochs))

        scheduler.step()

        # train for one epoch
        train_log = train(args, train_loader, model,
                          metric_fc, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(args, test_loader, model, metric_fc, criterion)

        print('loss %.4f - acc@1 %.4f - acc@5 %.4f - val_loss %.4f - val_acc@1 %.4f - val_acc@5 %.4f'
            %(train_log['loss'], train_log['acc@1'], train_log['acc@5'], val_log['loss'], val_log['acc@1'], val_log['acc@5']))

        tmp = pd.Series([
            epoch,
            scheduler.get_lr()[0],
            train_log['loss'],
            train_log['acc@1'],
            train_log['acc@5'],
            val_log['loss'],
            val_log['acc@1'],
            val_log['acc@5'],
        ], index=['epoch', 'lr', 'loss', 'acc@1', 'acc@5', 'val_loss', 'val_acc1', 'val_acc5'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/%s/log.csv' % args.name, index=False)

        if val_log['loss'] < best_loss:
            torch.save(model.state_dict(), 'models/%s/model.pth' %args.name)
            best_loss = val_log['loss']
            print("=> saved best model")


if __name__ == '__main__':
    main()
