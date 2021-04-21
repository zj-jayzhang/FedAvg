#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import copy
import os
import shutil
import warnings

import numpy as np
from tqdm import tqdm

warnings.filterwarnings('ignore')
import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate
from models import CNNMnist, CNNCifar
from utils import get_dataset, average_weights, exp_details
import torch.nn.functional as F


def inference(model, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()
            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    return acc, test_loss


def prepare_folders(cur_path):
    folders_util = [
        os.path.join(cur_path + '/logs', args.store_name),
        os.path.join(cur_path + '/checkpoints', args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


def save_checkpoint(state, is_best):
    filename = '{}/{}/ckpt.pth.tar'.format(os.path.abspath(os.path.dirname(os.getcwd())) + '/checkpoints',
                                           args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


if __name__ == '__main__':

    args = args_parser()
    args.type = 'iid' if args.iid == 1 else 'non-iid'
    args.store_name = '_'.join(
        [args.dataset, args.model, args.type, 'lr-' + str(args.lr)])
    cur_path = os.path.abspath(os.path.dirname(os.getcwd()))
    prepare_folders(cur_path)
    exp_details(args)

    logger_file = open(os.path.join(cur_path + '/logs', args.store_name, 'log.txt'), 'w')
    tf_writer = SummaryWriter(log_dir=os.path.join(cur_path + '/logs', args.store_name))

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128,
                                              shuffle=False, num_workers=4)
    # BUILD MODEL
    if args.dataset == 'mnist':
        global_model = CNNMnist(args).cuda()
    elif args.dataset == 'cifar':
        global_model = CNNCifar(args).cuda()

    bst_acc = -1
    description = "inference acc={:.4f}% loss={:.2f}, best_acc = {:.2f}%"
    for epoch in tqdm(range(args.epochs)):
        local_weights = []
        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx])
            w = local_model.update_weights(
                model=copy.deepcopy(global_model))
            local_weights.append(copy.deepcopy(w))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        test_acc, test_loss = inference(global_model, test_loader)

        tf_writer.add_scalar('test_acc', test_acc, epoch)
        tf_writer.add_scalar('test_loss', test_loss, epoch)

        output_log = 'After {} global rounds, Test acc: {}, inference loss: {}'.format(
            epoch + 1, test_acc, test_loss)

        logger_file.write(output_log + '\n')
        logger_file.flush()

        is_best = test_acc > bst_acc
        bst_acc = max(bst_acc, test_acc)
        print(description.format(test_acc, test_loss, bst_acc))
        
        save_checkpoint(global_model.state_dict(), is_best)

"""
python3 federated_main.py --model=cnn --dataset=cifar --iid=1 --epochs=300 --lr=0.01 --local_ep=5 --local_bs=32

python3 federated_main.py --model=cnn --dataset=mnist --iid=1 --epochs=100 --lr=0.01 --local_ep=5 --local_bs=32

"""