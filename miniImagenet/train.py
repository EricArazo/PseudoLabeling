import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch import optim
from torchvision import datasets, transforms

import numpy as np
import random
import sys
import argparse
import os
import time

from dataset.miniImagenet import get_dataset

sys.path.append('../utils_pseudoLab/')
from TwoSampler import *
from utils_ssl import *

from ssl_networks import resnet18
from ssl_networks import resnet18_wndrop
from ssl_networks import CNN as MT_Net

def parse_args():
    parser = argparse.ArgumentParser(description='command for the first train')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=100, help='Number of images in each mini-batch')
    parser.add_argument('--test_batch_size', type=int, default=100, help='Number of images in each mini-batch')
    parser.add_argument('--epoch', type=int, default=150, help='Training epoches')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--dataset_type', default='ssl', help='How to prepare the data: only labeled data for the warmUp ("ssl_warmUp") or unlabeled and labeled for the SSL training ("ssl")')
    parser.add_argument('--train_root', default='./data/miniImagenet/miniImagenet84', help='Root for train data')
    parser.add_argument('--labeled_samples', type=int, default=10000, help='Number of labeled samples')
    parser.add_argument('--reg1', type=float, default=0.8, help='Hyperparam for loss')
    parser.add_argument('--reg2', type=float, default=0.4, help='Hyperparam for loss')
    parser.add_argument('--download', type=bool, default=False, help='Download dataset')
    parser.add_argument('--network', type=str, default='resnet18_wndrop', help='The backbone of the network')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--seed_val', type=int, default=1, help='Seed for the validation split')
    parser.add_argument('--M', action='append', type=int, default=[], help="Milestones for the LR sheduler")
    parser.add_argument('--experiment_name', type=str, default = 'Proof',help='Name of the experiment (for the output files)')
    parser.add_argument('--loss_term', type=str, default='MixUp_ep', help='The regularizatio to use: "None", "Reg_e", "Reg_p", "Reg_ep", or "Reg_d"')
    parser.add_argument('--num_classes', type=int, default=100, help='Beta parameter for the EMA in the soft labels')
    parser.add_argument('--dropout', type=float, default=0.0, help='CNN dropout')
    parser.add_argument('--load_epoch', type=int, default=0, help='Load model from the last epoch from the warmup')
    parser.add_argument('--Mixup_Alpha', type=float, default=1, help='Alpha value for the beta dist from mixup')
    parser.add_argument('--cuda_dev', type=int, default=0, help='Set to 1 to choose the second gpu')
    parser.add_argument('--dataset', type=str, default='miniImagenet', help='Dataset name')
    parser.add_argument('--swa', type=str, default='True', help='Apply SWA')
    parser.add_argument('--swa_start', type=int, default=350, help='Start SWA')
    parser.add_argument('--swa_freq', type=float, default=5, help='Frequency')
    parser.add_argument('--swa_lr', type=float, default=0.001, help='LR')
    parser.add_argument('--labeled_batch_size', default=16, type=int, metavar='N', help="Labeled examples per minibatch (default: no constrain)")
    parser.add_argument('--validation_exp', type=str, default='False', help='Ignore the testing set during training and evaluation (it gets 5k samples from the training data to do the validation step)')
    parser.add_argument('--val_samples', type=int, default=0, help='Number of samples to be kept for validation (from the training set))')
    parser.add_argument('--pre_load', type=str, default='False', help='Load all the images to memory')
    parser.add_argument('--DA', type=str, default='standard', help='Chose the type of DA')
    parser.add_argument('--DApseudolab', type=str, default="True", help='Apply data augmentation when computing pseudolabels')
    parser.add_argument('--drop_extra_forward', type=str, default='True', help='Do an extra forward pass to compute the labels without dropout.')


    args = parser.parse_args()
    return args

def data_config(args, transform_train, transform_test):

    ####################################### Train ##########################################################
    trainset, unlabeled_indexes, labeled_indexes, valset = get_dataset(args, transform_train, transform_test)

    if args.labeled_batch_size > 0 and not args.dataset_type == 'ssl_warmUp':
        print("Training with two samplers. {0} clean samples per batch".format(args.labeled_batch_size))
        batch_sampler = TwoStreamBatchSampler(unlabeled_indexes, labeled_indexes, args.batch_size, args.labeled_batch_size)
        train_loader = torch.utils.data.DataLoader(trainset, batch_sampler=batch_sampler, num_workers=0, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # Only implmemnted the SOTA experiments (no hyperparameter choosing in MiniImageNet)

    test_loader = torch.utils.data.DataLoader(valset, batch_size=args.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # train and val
    print('-------> Data loading')
    print("Training with {0} labeled samples ({1} unlabeled samples)".format(len(labeled_indexes), len(unlabeled_indexes)))
    return train_loader, test_loader, unlabeled_indexes


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def main(args):
    best_ac = 0.0

    #####################
    # Initializing seeds and preparing GPU
    if args.cuda_dev == 1:
        torch.cuda.set_device(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.manual_seed(args.seed)  # CPU seed
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)  # GPU seed
    random.seed(args.seed)  # python seed for image transformation
    np.random.seed(args.seed)
    #####################

    if args.dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif args.dataset == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    elif args.dataset == 'miniImagenet':
        mean = [0.4728, 0.4487, 0.4031]
        std = [0.2744, 0.2663 , 0.2806]

    if args.DA == "standard":
        transform_train = transforms.Compose([
            transforms.Pad(6, padding_mode='reflect'),
            transforms.RandomCrop(84),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    elif args.DA == "jitter":
        transform_train = transforms.Compose([
            transforms.Pad(6, padding_mode='reflect'),
            transforms.ColorJitter(brightness= 0.4, contrast= 0.4, saturation= 0.4, hue= 0.1),
            transforms.RandomCrop(84),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        print("Wrong value for --DA argument.")


    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # data lodaer
    train_loader, test_loader, unlabeled_indexes = data_config(args, transform_train, transform_test)

    if args.network == "TE_Net":
        print("Loading TE_Net...")
        model = TE_Net(num_classes = args.num_classes).to(device)

    elif args.network == "MT_Net":
        print("Loading MT_Net...")
        model = MT_Net(num_classes = args.num_classes).to(device)

    elif args.network == "resnet18":
        print("Loading Resnet18...")
        model = resnet18(num_classes = args.num_classes).to(device)

    elif args.network == "resnet18_wndrop":
        print("Loading Resnet18...")
        model = resnet18_wndrop(num_classes = args.num_classes).to(device)


    print('Total params: {:.2f} M'.format((sum(p.numel() for p in model.parameters()) / 1000000.0)))

    milestones = args.M

    if args.swa == 'True':
        # to install it:
        # pip3 install torchcontrib
        # git clone https://github.com/pytorch/contrib.git
        # cd contrib
        # sudo python3 setup.py install
        from torchcontrib.optim import SWA
        base_optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        optimizer = SWA(base_optimizer, swa_lr=args.swa_lr)

    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    loss_train_epoch = []
    loss_val_epoch = []
    acc_train_per_epoch = []
    acc_val_per_epoch = []
    new_labels = []

    exp_path = os.path.join('./', 'ssl_models_{0}'.format(args.experiment_name), str(args.labeled_samples))
    res_path = os.path.join('./', 'metrics_{0}'.format(args.experiment_name), str(args.labeled_samples))

    if not os.path.isdir(res_path):
        os.makedirs(res_path)

    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)

    cont = 0
    load = False
    save = True

    if args.load_epoch != 0:
        load_epoch = args.load_epoch
        load = True
        save = False

    if args.dataset_type == 'ssl_warmUp':
        load = False
        save = True

    if load:
        if args.loss_term == 'Reg_ep':
            train_type = 'C'
        if args.loss_term == 'MixUp_ep':
            train_type = 'M'
        path = './checkpoints/warmUp_{0}_{1}_{2}_{3}_{4}_{5}_S{6}.hdf5'.format(train_type, \
                                                                                args.Mixup_Alpha, \
                                                                                load_epoch, \
                                                                                args.dataset, \
                                                                                args.labeled_samples, \
                                                                                args.network, \
                                                                                args.seed)

        checkpoint = torch.load(path)
        print("Load model in epoch " + str(checkpoint['epoch']))
        print("Path loaded: ", path)
        model.load_state_dict(checkpoint['state_dict'])
        print("Relabeling the unlabeled samples...")
        model.eval()
        results = np.zeros((len(train_loader.dataset), args.num_classes), dtype=np.float32)
        for images, images_pslab, labels, soft_labels, index in train_loader:

            images = images.to(device)
            labels = labels.to(device)
            soft_labels = soft_labels.to(device)

            outputs = model(images)
            prob, loss = loss_soft_reg_ep(outputs, labels, soft_labels, device, args)
            results[index.detach().numpy().tolist()] = prob.cpu().detach().numpy().tolist()

        train_loader.dataset.update_labels(results, unlabeled_indexes)
        print("Start training...")

    ####################################################################################################
    ###############################               TRAINING                ##############################
    ####################################################################################################

    for epoch in range(1, args.epoch + 1):
        st = time.time()
        scheduler.step()
        # train for one epoch
        print(args.experiment_name, args.labeled_samples)

        loss_per_epoch_train, \
        top_5_train_ac, \
        top1_train_ac, \
        train_time = train_CrossEntropy(args, model, device, \
                                        train_loader, optimizer, \
                                        epoch, unlabeled_indexes)

        loss_train_epoch += [loss_per_epoch_train]

        loss_per_epoch_test, acc_val_per_epoch_i = testing(args, model, device, test_loader)

        loss_val_epoch += loss_per_epoch_test
        acc_train_per_epoch += [top1_train_ac]
        acc_val_per_epoch += acc_val_per_epoch_i


        ####################################################################################################
        #############################               SAVING MODELS                ###########################
        ####################################################################################################
        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoints')

        if epoch == 1:
            best_acc_val = acc_val_per_epoch_i[-1]
            snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_labels_%d_bestAccVal_%.5f' % (
                epoch, loss_per_epoch_test[-1], acc_val_per_epoch_i[-1], args.labeled_samples, best_acc_val)
            torch.save(model.state_dict(), os.path.join(exp_path, snapBest + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapBest + '.pth'))
        else:
            if acc_val_per_epoch_i[-1] > best_acc_val:
                best_acc_val = acc_val_per_epoch_i[-1]

                if cont > 0:
                    try:
                        os.remove(os.path.join(exp_path, 'opt_' + snapBest + '.pth'))
                        os.remove(os.path.join(exp_path, snapBest + '.pth'))
                    except OSError:
                        pass
                snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_labels_%d_bestAccVal_%.5f' % (
                    epoch, loss_per_epoch_test[-1], acc_val_per_epoch_i[-1], args.labeled_samples, best_acc_val)
                torch.save(model.state_dict(), os.path.join(exp_path, snapBest + '.pth'))
                torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapBest + '.pth'))

        cont += 1

        if epoch == args.epoch:
            snapLast = 'last_epoch_%d_valLoss_%.5f_valAcc_%.5f_labels_%d_bestValLoss_%.5f' % (
                epoch, loss_per_epoch_test[-1], acc_val_per_epoch_i[-1], args.labeled_samples, best_acc_val)
            torch.save(model.state_dict(), os.path.join(exp_path, snapLast + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapLast + '.pth'))

        ### Saving model to load it again
        # cond = epoch%1 == 0
        if args.dataset_type == 'ssl_warmUp':
            if args.loss_term == 'Reg_ep':
                train_type = 'C'
            if args.loss_term == 'MixUp_ep':
                train_type = 'M'

            cond = (epoch==args.epoch)
            name = 'warmUp_{1}_{0}'.format(args.Mixup_Alpha, train_type)
            save = True
        else:
            cond = False

        if cond and save:
            print("Saving models...")
            path = './checkpoints/{0}_{1}_{2}_{3}_{4}_S{5}.hdf5'.format(name, epoch, args.dataset, \
                                                                        args.labeled_samples, \
                                                                        args.network, \
                                                                        args.seed)
            save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'loss_train_epoch' : np.asarray(loss_train_epoch),
                    'loss_val_epoch' : np.asarray(loss_val_epoch),
                    'acc_train_per_epoch' : np.asarray(acc_train_per_epoch),
                    'acc_val_per_epoch' : np.asarray(acc_val_per_epoch),
                    'labels': np.asarray(train_loader.dataset.soft_labels)

                }, filename = path)

        ####################################################################################################
        ############################               SAVING METRICS                ###########################
        ####################################################################################################

        # Save losses:
        np.save(res_path + '/' + str(args.labeled_samples) + '_LOSS_epoch_train.npy', np.asarray(loss_train_epoch))
        np.save(res_path + '/' + str(args.labeled_samples) + '_LOSS_epoch_val.npy', np.asarray(loss_val_epoch))

        # save accuracies:
        np.save(res_path + '/' + str(args.labeled_samples) + '_accuracy_per_epoch_train.npy',np.asarray(acc_train_per_epoch))
        np.save(res_path + '/' + str(args.labeled_samples) + '_accuracy_per_epoch_val.npy', np.asarray(acc_val_per_epoch))

    # applying swa
    if args.swa == 'True':
        optimizer.swap_swa_sgd()
        optimizer.bn_update(train_loader, model, device)
        loss_swa, acc_val_swa = testing(args, model, device, test_loader)

        snapLast = 'last_epoch_%d_valLoss_%.5f_valAcc_%.5f_labels_%d_bestValLoss_%.5f_swaAcc_%.5f' % (
            epoch, loss_per_epoch_test[-1], acc_val_per_epoch_i[-1], args.labeled_samples, best_acc_val, acc_val_swa[0])
        torch.save(model.state_dict(), os.path.join(exp_path, snapLast + '.pth'))
        torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapLast + '.pth'))

    print('Best ac:%f' % best_acc_val)


if __name__ == "__main__":
    args = parse_args()
    # train
    main(args)
