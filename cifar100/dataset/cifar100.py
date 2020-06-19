import torchvision as tv
import numpy as np
from PIL import Image
import time


def get_dataset(args, transform_train, transform_val):
    # prepare datasets
    cifar100_train_val = tv.datasets.CIFAR100(args.train_root, train=True, download=args.download)

    # get train/val dataset
    train_indexes, val_indexes = train_val_split(args, cifar100_train_val.train_labels)
    train = Cifar100Train(args, train_indexes, train=True, transform=transform_train, pslab_transform = transform_val)
    validation = Cifar100Train(args, val_indexes, train=True, transform=transform_val, pslab_transform = transform_val)

    if args.dataset_type == 'ssl_warmUp':
        unlabeled_indexes, labeled_indexes = train.prepare_data_ssl_warmUp()
    elif args.dataset_type == 'ssl':
        unlabeled_indexes, labeled_indexes = train.prepare_data_ssl()

    return train, unlabeled_indexes, labeled_indexes, validation


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
    def __init__(self, args, train_indexes=None, train=True, transform=None, target_transform=None, pslab_transform=None, download=False):
        super(Cifar100Train, self).__init__(args.train_root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.args = args
        if train_indexes is not None:
            self.train_data = self.train_data[train_indexes]
            self.train_labels = np.array(self.train_labels)[train_indexes]
        self.soft_labels = np.zeros((len(self.train_labels), 100), dtype=np.float32)
        self._num = int(len(self.train_labels) - int(args.labeled_samples))
        self.original_labels = np.copy(self.train_labels)
        self.pslab_transform = pslab_transform

    def prepare_data_ssl(self):
        np.random.seed(self.args.seed)
        original_labels = np.copy(self.train_labels)
        unlabeled_indexes = [] # initialize the vector
        labeled_indexes = []

        num_unlab_samples = self._num
        num_labeled_samples = len(self.train_labels) - num_unlab_samples

        labeled_per_class = int(num_labeled_samples / self.args.num_classes)
        unlab_per_class = int(num_unlab_samples / self.args.num_classes)

        for id in range(self.args.num_classes):
            indexes = np.where(original_labels == id)[0]
            np.random.shuffle(indexes)

            for i in range(len(indexes)):
                if i < unlab_per_class:
                    label_sym = np.random.randint(self.args.num_classes, dtype=np.int32)
                    self.train_labels[indexes[i]] = label_sym

                self.soft_labels[indexes[i]][self.train_labels[indexes[i]]] = 1

            unlabeled_indexes.extend(indexes[:unlab_per_class])
            labeled_indexes.extend(indexes[unlab_per_class:])

        return np.asarray(unlabeled_indexes),  np.asarray(labeled_indexes)

    def prepare_data_ssl_warmUp(self):
        # to be more equal, every category can be processed separately
        # np.random.seed(42)
        np.random.seed(self.args.seed)

        original_labels = np.copy(self.train_labels)
        unlabeled_indexes = [] # initialize the vector
        train_indexes = []

        num_unlab_samples = self._num
        num_labeled_samples = len(self.train_labels) - num_unlab_samples

        labeled_per_class = int(num_labeled_samples / self.args.num_classes)
        unlab_per_class = int(num_unlab_samples / self.args.num_classes)

        for id in range(self.args.num_classes):
            indexes = np.where(original_labels == id)[0]
            np.random.shuffle(indexes)

            unlabeled_indexes.extend(indexes[:unlab_per_class])
            train_indexes.extend(indexes[unlab_per_class:])

        np.asarray(train_indexes)

        ### Redefining variables with the new number of training examples
        self.train_data = self.train_data[train_indexes]
        self.train_labels = np.array(self.train_labels)[train_indexes]
        self.soft_labels = np.zeros((len(self.train_labels), self.args.num_classes), dtype=np.float32)

        for i in range(len(self.train_data)):
            self.soft_labels[i][self.train_labels[i]] = 1

        unlabeled_indexes = np.asarray([])

        return np.asarray(unlabeled_indexes), np.asarray(train_indexes)


    def update_labels(self, result, unlabeled_indexes):
        relabel_indexes = list(unlabeled_indexes)

        self.soft_labels[relabel_indexes] = result[relabel_indexes]
        self.train_labels[relabel_indexes] = self.soft_labels[relabel_indexes].argmax(axis = 1).astype(np.int64)

        print("Samples relabeled with the prediction: ", str(len(relabel_indexes)))


    def __getitem__(self, index):
        img, labels, soft_labels = self.train_data[index], self.train_labels[index], self.soft_labels[index]
        img = Image.fromarray(img)

        if self.args.DApseudolab == "False":
            img_pseudolabels = self.pslab_transform(img)
        else:
            img_pseudolabels = 0

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            labels = self.target_transform(labels)

        return img, img_pseudolabels, labels, soft_labels, index
