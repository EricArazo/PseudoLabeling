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
import logging
import os
import time
from IPython import embed


from dataset.cifar100 import get_dataset

sys.path.append('../utils_pseudoLab/')
from TwoSampler import *
from utils_noise import *
from ssl_networks import CNN as MT_Net

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description='command for the first train')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=100, help='#images in each mini-batch')
    parser.add_argument('--test_batch_size', type=int, default=100, help='#images in each mini-batch')
    parser.add_argument('--epoch', type=int, default=150, help='training epoches')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--dataset_type', default='semiSup', help='noise type of the dataset')
    parser.add_argument('--train_root', default='./data', help='root for train data')
    parser.add_argument('--epoch_begin', default=0, type=int, help='the epoch to begin update labels')
    parser.add_argument('--epoch_update', default=1, type=int, help='#epoch to average to update soft labels')
    parser.add_argument('--labeled_samples', type=int, default=10000, help='number of labeled samples')
    parser.add_argument('--out', type=str, default='./data/model_data', help='Directory of the output')
    parser.add_argument('--alpha', type=float, default=0.8, help='Hyper param for loss')
    parser.add_argument('--beta', type=float, default=0.4, help='Hyper param for loss')
    parser.add_argument('--download', type=bool, default=False, help='download dataset')
    parser.add_argument('--network', type=str, default='MT_Net', help='the backbone of the network')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--seed_val', type=int, default=1, help='seed for the validation split')
    parser.add_argument('--M', action='append', type=int, default=[], help="Milestones for the LR sheduler")
    parser.add_argument('--experiment_name', type=str, default = 'Proof',help='name of the experiment (for the output files)')
    parser.add_argument('--label_noise', type=float, default = 0.0,help='ratio of labeles to relabel randomly')
    parser.add_argument('--loss_term', type=str, default='MixUp_ep', help='the regularizatio to use: "None", "Reg_e", "Reg_p", "Reg_ep", or "Reg_d"')
    parser.add_argument('--relab', type=str, default='unifRelab', help='choose how to relabel the random samples from the unlabeled set')
    parser.add_argument('--num_classes', type=int, default=100, help='beta parameter for the EMA in the soft labels')
    parser.add_argument('--gausTF', type=bool, default=False, help='apply gaussian noise')
    parser.add_argument('--dropout', type=float, default=0.0, help='cnn dropout')
    parser.add_argument('--initial_epoch', type=int, default=0, help='#images in each mini-batch')
    parser.add_argument('--Mixup_Alpha', type=float, default=1, help='alpha value for the beta dist from mixup')
    parser.add_argument('--cuda_dev', type=int, default=0, help='set to 1 to choose the second gpu')
    parser.add_argument('--save_checkpoint', type=str, default= "False", help='save checkpoints for ensembles')
    parser.add_argument('--dataset', type=str, default='cifar100', help='Daraset name')
    parser.add_argument('--swa', type=str, default='True', help='Apply SWA')
    parser.add_argument('--swa_start', type=int, default=350, help='Start SWA')
    parser.add_argument('--swa_freq', type=float, default=5, help='Frequency')
    parser.add_argument('--swa_lr', type=float, default=0.001, help='LR')
    parser.add_argument('--labeled_batch_size', default=8, type=int, metavar='N', help="labeled examples per minibatch (default: no constrain)")
    parser.add_argument('--validation_exp', type=str, default='False', help='Ignore the testing set during training and evaluation (it gets 5k samples from the training data to do the validation step)')
    parser.add_argument('--val_samples', type=int, default=5000, help='Number of samples to be kept for validation (from the training set))')

    args = parser.parse_args()
    return args

def data_config(args, transform_train, transform_test, dst_folder):

    ####################################### Train ##########################################################
    trainset, clean_labels, noisy_labels, train_noisy_indexes, train_clean_indexes, valset = get_dataset(args, transform_train, transform_test, dst_folder)

    if args.labeled_batch_size > 0 and not args.dataset_type == 'sym_noise_warmUp':
        print("Training with two samplers. {0} clean samples per batch".format(args.labeled_batch_size))
        batch_sampler = TwoStreamBatchSampler(train_noisy_indexes, train_clean_indexes, args.batch_size, args.labeled_batch_size)
        train_loader = torch.utils.data.DataLoader(trainset, batch_sampler=batch_sampler, num_workers=0, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)


    if args.validation_exp == "True":
        print("Training to choose hyperparameters --- VALIDATON MODE ---.")
        testset = valset
    else:
        print("Training to compare to the SOTA --- TESTING MODE ---.")
        testset = datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # train and val
    print('-------> Data loading')
    print("Training with {0} labeled samples ({1} unlabeled samples)".format(len(clean_labels)-len(train_noisy_indexes), len(train_noisy_indexes)))
    return train_loader, test_loader, train_noisy_indexes




def record_params(args):
    dst_folder = args.out + '/' + args.dataset_type + '-lr-{}-ratio-{}-alpha-{}-beta-{}-{}'.format(args.lr,
                                                                                                   args.labeled_samples,
                                                                                                   args.alpha,
                                                                                                   args.beta,
                                                                                                   args.network)
    dst_folder = dst_folder + '/first_train'
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    rd = open(dst_folder + '/config.txt', 'w')
    rd.write('lr:%f' % args.lr + '\n')
    rd.write('wd:%f' % args.wd + '\n')
    rd.write('momentum:%f' % args.momentum + '\n')
    rd.write('batch_size:%d' % args.batch_size + '\n')
    rd.write('epoch:%d' % args.epoch + '\n')
    rd.write('dataset_type:' + args.dataset_type + '\n')
    rd.write('labeled_samples:%f' % args.labeled_samples + '\n')
    rd.close()

    handler = logging.FileHandler(dst_folder + '/train.log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return dst_folder


def record_result(dst_folder, best_ac):
    dst = dst_folder + '/config.txt'
    rd = open(dst, 'a+')
    rd.write('first_train:best_ac:%.3f' % best_ac + '\n')
    rd.close()

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def main(args, dst_folder):
    # best_ac only record the best top1_ac for validation set.
    best_ac = 0.0
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if args.cuda_dev == 1:
        torch.cuda.set_device(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.manual_seed(args.seed)  # CPU seed
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)  # GPU seed

    random.seed(args.seed)  # python seed for image transformation
    np.random.seed(args.seed)

    if args.dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif args.dataset == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]

    transform_train = transforms.Compose([
        transforms.Pad(2, padding_mode='reflect'),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # data lodaer
    train_loader, test_loader, train_noisy_indexes = data_config(args, transform_train, transform_test,  dst_folder)


    if args.network == "MT_Net":
        print("Loading MT_Net...")
        model = MT_Net(num_classes = args.num_classes, dropRatio = args.dropout).to(device)

    #For multiple GPUs
    #model = torch.nn.DataParallel(resnet18()).to(device)


    print('Total params: %2.fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    milestones = args.M

    if args.swa == 'True':
        # to install it:
        # pip3 install torchcontrib
        # git clone https://github.com/pytorch/contrib.git
        # cd contrib
        # sudo python3 setup.py install
        from torchcontrib.optim import SWA
        base_optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
        optimizer = SWA(base_optimizer, swa_lr=args.swa_lr)

    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    loss_train_epoch = []
    loss_val_epoch = []
    acc_train_per_epoch = []
    acc_val_per_epoch = []
    new_labels = []


    exp_path = os.path.join('./', 'noise_models_{0}'.format(args.experiment_name), str(args.labeled_samples))
    res_path = os.path.join('./', 'metrics_{0}'.format(args.experiment_name), str(args.labeled_samples))

    if not os.path.isdir(res_path):
        os.makedirs(res_path)

    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)

    cont = 0

    load = False
    save = True

    if args.initial_epoch != 0:
        initial_epoch = args.initial_epoch
        load = True
        save = False

    if args.dataset_type == 'sym_noise_warmUp':
        load = False
        save = True

    if load:
        if args.loss_term == 'Reg_ep':
            train_type = 'C'
        if args.loss_term == 'MixUp_ep':
            train_type = 'M'
        if args.dropout > 0.0:
            train_type = train_type + 'drop' + str(int(10*args.dropout))
        path = './checkpoints/warmUp_{6}_{5}_{0}_{1}_{2}_{3}_S{4}.hdf5'.format(initial_epoch, \
                                                                                args.dataset, \
                                                                                args.labeled_samples, \
                                                                                args.network, \
                                                                                args.seed, \
                                                                                args.Mixup_Alpha, \
                                                                                train_type)
        checkpoint = torch.load(path)
        print("Load model in epoch " + str(checkpoint['epoch']))
        print("Path loaded: ", path)
        model.load_state_dict(checkpoint['state_dict'])
        print("Relabeling the unlabeled samples...")
        model.eval()
        initial_rand_relab = args.label_noise
        results = np.zeros((len(train_loader.dataset), args.num_classes), dtype=np.float32)
        for images, labels, soft_labels, index in train_loader:

            images = images.to(device)
            labels = labels.to(device)
            soft_labels = soft_labels.to(device)

            outputs = model(images)
            prob, loss = loss_soft_reg_ep(outputs, labels, soft_labels, device, args)
            results[index.detach().numpy().tolist()] = prob.cpu().detach().numpy().tolist()

        train_loader.dataset.update_labels_randRelab(results, train_noisy_indexes, initial_rand_relab)
        print("Start training...")

    for epoch in range(1, args.epoch + 1):
        st = time.time()
        scheduler.step()
        # train for one epoch
        print(args.experiment_name, args.labeled_samples)

        loss_per_epoch, top_5_train_ac, top1_train_acc_original_labels, \
        top1_train_ac, train_time = train_CrossEntropy_partialRelab(\
                                                        args, model, device, \
                                                        train_loader, optimizer, \
                                                        epoch, train_noisy_indexes)
        loss_train_epoch += [loss_per_epoch]

        # test
        if args.validation_exp == "True":
            loss_per_epoch, acc_val_per_epoch_i = validating(args, model, device, test_loader)
        else:
            loss_per_epoch, acc_val_per_epoch_i = testing(args, model, device, test_loader)

        loss_val_epoch += loss_per_epoch
        acc_train_per_epoch += [top1_train_ac]
        acc_val_per_epoch += acc_val_per_epoch_i


        ####################################################################################################
        #############################               SAVING MODELS                ###########################
        ####################################################################################################


        if epoch == 1:
            best_acc_val = acc_val_per_epoch_i[-1]
            snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%d_bestAccVal_%.5f' % (
                epoch, loss_per_epoch[-1], acc_val_per_epoch_i[-1], args.labeled_samples, best_acc_val)
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
                snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%d_bestAccVal_%.5f' % (
                    epoch, loss_per_epoch[-1], acc_val_per_epoch_i[-1], args.labeled_samples, best_acc_val)
                torch.save(model.state_dict(), os.path.join(exp_path, snapBest + '.pth'))
                torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapBest + '.pth'))

        cont += 1

        if epoch == args.epoch:
            snapLast = 'last_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%d_bestValLoss_%.5f' % (
                epoch, loss_per_epoch[-1], acc_val_per_epoch_i[-1], args.labeled_samples, best_acc_val)
            torch.save(model.state_dict(), os.path.join(exp_path, snapLast + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapLast + '.pth'))


        #### Save models for ensembles:
        if (epoch >= 150) and (epoch%2 == 0) and (args.save_checkpoint == "True"):
            print("Saving model ...")
            out_path = './checkpoints/ENS_{0}_{1}'.format(args.experiment_name, args.labeled_samples)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            torch.save(model.state_dict(), out_path + "/epoch_{0}.pth".format(epoch))

        ### Saving model to load it again
        # cond = epoch%1 == 0
        if args.dataset_type == 'sym_noise_warmUp':
            if args.loss_term == 'Reg_ep':
                train_type = 'C'
            if args.loss_term == 'MixUp_ep':
                train_type = 'M'
            if args.dropout > 0.0:
                train_type = train_type + 'drop' + str(int(10*args.dropout))

            cond = (epoch==args.epoch)
            name = 'warmUp_{1}_{0}'.format(args.Mixup_Alpha, train_type)
            save = True
        else:
            cond = False

        if cond and save:
            print("Saving models...")
            path = './checkpoints/{0}_{1}_{2}_{3}_{4}_S{5}.hdf5'.format(name, epoch, args.dataset, args.labeled_samples, args.network, args.seed)

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
        np.save(res_path + '/' + str(args.labeled_samples) + '_accuracy_per_epoch_train.npy',
                np.asarray(acc_train_per_epoch))
        np.save(res_path + '/' + str(args.labeled_samples) + '_accuracy_per_epoch_val.npy', np.asarray(acc_val_per_epoch))

        # save the new labels
        new_labels.append(train_loader.dataset.train_labels)
        np.save(res_path + '/' + str(args.labeled_samples) + '_new_labels.npy',
                np.asarray(new_labels))


        logging.info('Epoch: [{}|{}], train_loss: {:.3f}, top1_train_ac: {:.3f}, top1_val_ac: {:.3f}, train_time: {:.3f}'.format(
            epoch, args.epoch, loss_per_epoch[-1], top1_train_ac, acc_val_per_epoch_i[-1], time.time() - st))

    # applying swa
    if args.swa == 'True':
        optimizer.swap_swa_sgd()
        optimizer.bn_update(train_loader, model, device)
        if args.validation_exp == "True":
            loss_swa, acc_val_swa = validating(args, model, device, test_loader)
        else:
            loss_swa, acc_val_swa = testing(args, model, device, test_loader)

        snapLast = 'last_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%d_bestValLoss_%.5f_swaAcc_%.5f' % (
            epoch, loss_per_epoch[-1], acc_val_per_epoch_i[-1], args.labeled_samples, best_acc_val, acc_val_swa[0])
        torch.save(model.state_dict(), os.path.join(exp_path, snapLast + '.pth'))
        torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapLast + '.pth'))

    # save_fig(dst_folder)
    print('Best ac:%f' % best_acc_val)
    record_result(dst_folder, best_ac)


if __name__ == "__main__":
    args = parse_args()
    logging.info(args)
    # record params
    dst_folder = record_params(args)
    # train
    main(args, dst_folder)
