import torchvision as tv
import numpy as np
from PIL import Image
from IPython import embed
import time
from torch.utils.data import Dataset
from os.path import join
import csv
from tqdm import tqdm


def get_dataset(args, transform_train, transform_val):
    # prepare datasets
    train_data, train_labels, val_data, val_labels = make_dataset(args)

    train = MiniImagenet84(args, train_data, train_labels, train=True, transform=transform_train, pslab_transform = transform_val)
    val = MiniImagenet84(args, val_data, val_labels, train=False, transform=transform_val, pslab_transform = transform_val)

    if args.dataset_type == 'ssl_warmUp':
        unlabeled_indexes, labeled_indexes = train.prepare_data_ssl_warmUp()
    elif args.dataset_type == 'ssl':
        unlabeled_indexes, labeled_indexes = train.prepare_data_ssl()

    return train, unlabeled_indexes, labeled_indexes, val

class MiniImagenet84(Dataset):
    def __init__(self, args, data, labels, train=True, transform=None, target_transform=None, pslab_transform = None, download=False):
        self.args = args
        self.train_data, self.train_labels =  data, labels
        self.soft_labels = np.zeros((len(self.train_labels), args.num_classes), dtype=np.float32)
        self._num = int(len(self.train_labels) - int(args.labeled_samples))
        self.original_labels = np.copy(self.train_labels)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.pslab_transform = pslab_transform

    def prepare_data_ssl(self):
        np.random.seed(self.args.seed)
        original_labels = np.copy(self.train_labels)
        unabeled_indexes = [] # initialize the vector
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

            unabeled_indexes.extend(indexes[:unlab_per_class])
            labeled_indexes.extend(indexes[unlab_per_class:])

        # print("Training with {0} labeled samples ({1} unlabeled samples)".format(num_clean_samples, num_unlab_samples))
        return np.asarray(unabeled_indexes),  np.asarray(labeled_indexes)


    def prepare_data_ssl_warmUp(self):
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

        ### Redefining variables with the new number oftraining examples
        self.train_data = self.train_data[train_indexes]
        self.train_labels = np.array(self.train_labels)[train_indexes]
        self.soft_labels = np.zeros((len(self.train_labels), self.args.num_classes), dtype=np.float32)

        for i in range(len(self.train_data)):
            self.soft_labels[i][self.train_labels[i]] = 1

        unlabeled_indexes = np.asarray([])

        return np.asarray(unlabeled_indexes), np.asarray(train_indexes)


    def update_labels(self, result, train_unlabeled_indexes):
        relabel_indexes = list(train_unlabeled_indexes)

        self.soft_labels[relabel_indexes] = result[relabel_indexes]
        self.train_labels[relabel_indexes] = self.soft_labels[relabel_indexes].argmax(axis = 1).astype(np.int64)

        print("Samples relabeled with the prediction: ", str(len(relabel_indexes)))


    def __getitem__(self, index):
        # img_path, labels, soft_labels, z_exp_labels = self.train_data[index], self.train_labels[index], self.soft_labels[index], self.z_exp_labels[index]
        img, labels, soft_labels = self.train_data[index], self.train_labels[index], self.soft_labels[index]

        if self.args.pre_load == "True":
            img = Image.fromarray(img)
        else:
            img = Image.open(img)#.convert('RGB')
        # img = Image.fromarray(img)

        if self.args.DApseudolab == "False":
            img_pseudolabels = self.pslab_transform(img)
        else:
            img_pseudolabels = 0

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            labels = self.target_transform(labels)

        if self.train:
            return img, img_pseudolabels, labels, soft_labels, index
        else:
            return img, labels

    def __len__(self):
        return len(self.train_data)


class Cifar100Val(tv.datasets.CIFAR100):
    def __init__(self, root, val_indexes, train=True, transform=None, target_transform=None, download=False):
        super(Cifar100Val, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.val_labels = np.array(self.train_labels)[val_indexes]
        self.val_data = self.train_data[val_indexes]


def make_dataset(args):
    np.random.seed(42)
    csv_files = ["train.csv", "val.csv", "test.csv"]
    img_paths = []
    labels = []
    for split in csv_files:
        in_csv_path = join(args.train_root, split)
        in_images_path = join(args.train_root, "images")

        with open(in_csv_path) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)
            for i,row in enumerate(csvreader):
                img_paths.append(join(in_images_path,row[0]))
                labels.append(row[1])

    mapping = {y: x for x, y in enumerate(np.unique(labels))}
    label_mapped = [mapping[i] for i in labels]

    # labels

    # split in train and validation:
    train_num = 50000
    val_num = 10000

    idxes = np.random.permutation(len(img_paths))

    img_paths = np.asarray(img_paths)[idxes]
    label_mapped = np.asarray(label_mapped)[idxes]

    train_img_paths = img_paths[:train_num]
    train_labels = label_mapped[:train_num]
    val_img_paths = img_paths[train_num:]
    val_labels = label_mapped[train_num:]


    if args.pre_load == "True":
        train_pil_images = []
        print("Loading Images in memory...")
        for i in tqdm(train_img_paths):
            train_pil_images.append(np.asarray(Image.open(i)))
        train_data = np.asarray(train_pil_images)

        val_pil_images = []
        for i in val_img_paths:
            val_pil_images.append(np.asarray(Image.open(i)))
        val_data = np.asarray(val_pil_images)
    else:
        train_data = train_img_paths
        val_data = val_img_paths

    return train_data, train_labels, val_data, val_labels
