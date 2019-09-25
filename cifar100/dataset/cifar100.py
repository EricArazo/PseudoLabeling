import torchvision as tv
import numpy as np
from PIL import Image
from IPython import embed
import time


def get_dataset(args, transform_train, transform_val, dst_folder):
    # prepare datasets
    cifar100_train_val = tv.datasets.CIFAR100(args.train_root, train=True, download=args.download)

    # get train/val dataset
    train_indexes, val_indexes = train_val_split(args, cifar100_train_val.train_labels)
    train = Cifar100Train(args, dst_folder, train_indexes, train=True, transform=transform_train, pslab_transform = transform_val)
    validation = Cifar100Train(args, dst_folder, val_indexes, train=True, transform=transform_val, pslab_transform = transform_val)

    if args.dataset_type == 'sym_noise_warmUp':
        clean_labels, noisy_labels, noisy_indexes, clean_indexes = train.symmetric_noise_warmUp_semisup()
    elif args.dataset_type == 'semiSup':
        clean_labels, noisy_labels, noisy_indexes, clean_indexes = train.symmetric_noise_for_semiSup()

    return train, clean_labels, noisy_labels, noisy_indexes, clean_indexes, validation


def train_val_split(args, train_val):

    np.random.seed(args.seed_val)
    train_val = np.array(train_val)
    train_indexes = []
    val_indexes = []
    val_num = int(args.val_samples / args.num_classes)

    for id in range(args.num_classes):
        indexes = np.where(train_val == id)[0]
        np.random.shuffle(indexes)
        val_indexes.extend(indexes[:val_num])
        train_indexes.extend(indexes[val_num:])
    np.random.shuffle(train_indexes)
    np.random.shuffle(val_indexes)

    return train_indexes, val_indexes


class Cifar100Train(tv.datasets.CIFAR100):
    def __init__(self, args, dst_folder, train_indexes=None, train=True, transform=None, target_transform=None, pslab_transform=None, download=False):
        super(Cifar100Train, self).__init__(args.train_root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.args = args
        if train_indexes is not None:
            self.train_data = self.train_data[train_indexes]
            self.train_labels = np.array(self.train_labels)[train_indexes]
        self.soft_labels = np.zeros((len(self.train_labels), 100), dtype=np.float32)
        self.prediction = np.zeros((self.args.epoch_update, len(self.train_data), 100), dtype=np.float32)
        self.z_exp_labels = np.zeros((len(self.train_labels), 100), dtype=np.float32)
        self.Z_exp_labels = np.zeros((len(self.train_labels), 100), dtype=np.float32)
        self.z_soft_labels = np.zeros((len(self.train_labels), 100), dtype=np.float32)
        self._num = int(len(self.train_labels) - int(args.labeled_samples))
        self._count = 0
        self.dst = dst_folder.replace('second','first') + '/labels.npz'
        self.alpha = 0.6
        self.gaus_noise = self.args.gausTF
        self.original_labels = np.copy(self.train_labels)
        self.pslab_transform = pslab_transform

    def symmetric_noise_for_semiSup(self):
        np.random.seed(self.args.seed)
        original_labels = np.copy(self.train_labels)
        noisy_indexes = [] # initialize the vector
        clean_indexes = []

        num_unlab_samples = self._num
        num_clean_samples = len(self.train_labels) - num_unlab_samples

        clean_per_class = int(num_clean_samples / self.args.num_classes)
        unlab_per_class = int(num_unlab_samples / self.args.num_classes)

        for id in range(self.args.num_classes):
            indexes = np.where(original_labels == id)[0]
            np.random.shuffle(indexes)

            for i in range(len(indexes)):
                if i < unlab_per_class:
                    label_sym = np.random.randint(self.args.num_classes, dtype=np.int32)
                    self.train_labels[indexes[i]] = label_sym

                self.soft_labels[indexes[i]][self.train_labels[indexes[i]]] = 1

            noisy_indexes.extend(indexes[:unlab_per_class])
            clean_indexes.extend(indexes[unlab_per_class:])

        return original_labels, self.train_labels,  np.asarray(noisy_indexes),  np.asarray(clean_indexes)

    def symmetric_noise_warmUp_semisup(self):
        # to be more equal, every category can be processed separately
        # np.random.seed(42)
        np.random.seed(self.args.seed)

        original_labels = np.copy(self.train_labels)
        noisy_indexes = [] # initialize the vector
        train_indexes = []

        num_unlab_samples = self._num
        num_clean_samples = len(self.train_labels) - num_unlab_samples

        clean_per_class = int(num_clean_samples / self.args.num_classes)
        unlab_per_class = int(num_unlab_samples / self.args.num_classes)

        for id in range(self.args.num_classes):
            indexes = np.where(original_labels == id)[0]
            np.random.shuffle(indexes)

            noisy_indexes.extend(indexes[:unlab_per_class])
            train_indexes.extend(indexes[unlab_per_class:])

        np.asarray(train_indexes)

        ### Redefining variables with the new number oftraining examples
        self.train_data = self.train_data[train_indexes]
        self.train_labels = np.array(self.train_labels)[train_indexes]
        self.soft_labels = np.zeros((len(self.train_labels), self.args.num_classes), dtype=np.float32)

        for i in range(len(self.train_data)):
            self.soft_labels[i][self.train_labels[i]] = 1

        self.prediction = np.zeros((self.args.epoch_update, len(self.train_data), self.args.num_classes), dtype=np.float32)
        self.z_exp_labels = np.zeros((len(self.train_labels), self.args.num_classes), dtype=np.float32)
        self.Z_exp_labels = np.zeros((len(self.train_labels), self.args.num_classes), dtype=np.float32)
        self.z_soft_labels = np.zeros((len(self.train_labels), self.args.num_classes), dtype=np.float32)

        noisy_indexes = np.asarray([])

        return original_labels[train_indexes], self.train_labels, np.asarray(noisy_indexes), np.asarray(train_indexes)


    def update_labels_randRelab(self, result, train_noisy_indexes, rand_ratio):

        idx = self._count % self.args.epoch_update
        self.prediction[idx,:] = result
        nb_noisy = len(train_noisy_indexes)
        nb_rand = int(nb_noisy*rand_ratio)
        idx_noisy_all = list(range(nb_noisy))
        idx_noisy_all = np.random.permutation(idx_noisy_all)

        idx_rand = idx_noisy_all[:nb_rand]
        idx_relab = idx_noisy_all[nb_rand:]

        if rand_ratio == 0.0:
            idx_relab = list(range(len(train_noisy_indexes)))
            idx_rand = []

        if self._count >= self.args.epoch_begin:

            relabel_indexes = list(train_noisy_indexes[idx_relab])
            self.soft_labels[relabel_indexes] = result[relabel_indexes]
            self.train_labels[relabel_indexes] = self.soft_labels[relabel_indexes].argmax(axis = 1).astype(np.int64)


            for idx_num in train_noisy_indexes[idx_rand]:
                new_soft = np.ones(self.args.num_classes)
                new_soft = new_soft*(1/self.args.num_classes)

                self.soft_labels[idx_num] = new_soft
                self.train_labels[idx_num] = self.soft_labels[idx_num].argmax(axis = 0).astype(np.int64)


            print("Samples relabeled with the prediction: ", str(len(idx_relab)))
            print("Samples relabeled with '{0}': ".format(self.args.relab), str(len(idx_rand)))

        self.Z_exp_labels = self.alpha * self.Z_exp_labels + (1. - self.alpha) * self.prediction[idx,:]
        self.z_exp_labels =  self.Z_exp_labels * (1. / (1. - self.alpha ** (self._count + 1)))

        self._count += 1

        # save params
        if self._count == self.args.epoch:
            np.savez(self.dst, data=self.train_data, hard_labels=self.train_labels, soft_labels=self.soft_labels)



    def gaussian(self, ins, mean, stddev):
        noise = ins.data.new(ins.size()).normal_(mean, stddev)
        return ins + noise

    def __getitem__(self, index):
        img, labels, soft_labels, z_exp_labels = self.train_data[index], self.train_labels[index], self.soft_labels[index], self.z_exp_labels[index]
        img = Image.fromarray(img)

        if self.args.DApseudolab == "False":
            img_pseudolabels = self.pslab_transform(img)
        else:
            img_pseudolabels = 0

        if self.transform is not None:
            img = self.transform(img)
            if self.gaus_noise:
                img = self.gaussian(img, 0.0, 0.15)

        if self.target_transform is not None:
            labels = self.target_transform(labels)

        return img, img_pseudolabels, labels, soft_labels, index
