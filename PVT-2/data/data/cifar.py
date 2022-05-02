from torchvision.datasets import CIFAR10, CIFAR100
from copy import deepcopy
import numpy as np
# from config import cifar_10_root, cifar_100_root
cifar_10_root = '/home/neuron/Datasets/CIFAR/'
cifar_100_root = '/home/neuron/Datasets/CIFAR/'
class CustomCIFAR10(CIFAR10):

    def __init__(self, *args, **kwargs):

        super(CustomCIFAR10, self).__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):

        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label
            # , uq_idx

class CustomCIFAR100(CIFAR100):

    def __init__(self, *args, **kwargs):
        super(CustomCIFAR100, self).__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label
            # , uq_idx

def subsample_dataset(dataset, idxs):

    dataset.data = dataset.data[idxs]
    dataset.targets = np.array(dataset.targets)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset

def subsample_classes(dataset, include_classes=(0, 1, 8, 9)):

    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset

def get_train_val_split(train_dataset, val_split=0.2):

    val_dataset = deepcopy(train_dataset)
    train_dataset = deepcopy(train_dataset)

    train_classes = np.unique(train_dataset.targets)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.targets == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    # Get training/validation datasets based on selected idxs
    train_dataset = subsample_dataset(train_dataset, train_idxs)
    val_dataset = subsample_dataset(val_dataset, val_idxs)

    return train_dataset, val_dataset

def get_equal_len_datasets(dataset1, dataset2):

    """
    Make two datasets the same length
    """

    if len(dataset1) > len(dataset2):

        rand_idxs = np.random.choice(range(len(dataset1)), size=(len(dataset2,)))
        subsample_dataset(dataset1, rand_idxs)

    elif len(dataset2) > len(dataset1):

        rand_idxs = np.random.choice(range(len(dataset2)), size=(len(dataset1,)))
        subsample_dataset(dataset2, rand_idxs)

    return dataset1, dataset2

def get_cifar10_datasets(train_transform, test_transform, num_train_classes=5, num_auxiliary_classes=2,
                       num_open_set_classes=3, balance_open_set_eval=False, split_train_val=True, seed=0):

    np.random.seed(seed)
    random_sequence = list(range(10))
    np.random.shuffle(random_sequence)
    train_classes = random_sequence[:num_train_classes]
    auxiliary_classes = random_sequence[num_train_classes:(num_train_classes + num_auxiliary_classes)]
    open_set_classes = random_sequence[(num_train_classes + num_auxiliary_classes):(
                num_train_classes + num_auxiliary_classes + num_open_set_classes)]

    # Init train dataset and subsample training classes
    train_dataset_whole = CustomCIFAR10(root=cifar_10_root, transform=train_transform, train=True)
    train_dataset_whole = subsample_classes(train_dataset_whole, include_classes=train_classes)

    # Split into training and validation sets
    train_dataset_split, val_dataset_split = get_train_val_split(train_dataset_whole)
    val_dataset_split.transform = test_transform

    if auxiliary_classes is not None:
        # aux_dataset_known = CustomCub2011(root=cub_root, transform=train_transform, train=True)
        aux_dataset_whole = CustomCIFAR10(root=cifar_10_root, transform=test_transform, train=False)
        aux_dataset_known = subsample_classes(aux_dataset_whole, include_classes=auxiliary_classes)
        train_dataset_whole3 = CustomCIFAR10(root=cifar_10_root, transform=train_transform, train=True)
        mix_dataset_train = subsample_classes(train_dataset_whole3, include_classes=train_classes+auxiliary_classes)
        train_dataset_whole4 = CustomCIFAR10(root=cifar_10_root, transform=test_transform, train=False)
        mix_dataset_test = subsample_classes(train_dataset_whole4, include_classes=train_classes+auxiliary_classes)

    # Get test set for known classes
    test_dataset_known = CustomCIFAR10(root=cifar_10_root, transform=test_transform, train=False)
    test_dataset_known = subsample_classes(test_dataset_known, include_classes=train_classes)

    # Get testset for unknown classes
    test_dataset_unknown = CustomCIFAR10(root=cifar_10_root, transform=test_transform, train=False)
    test_dataset_unknown = subsample_classes(test_dataset_unknown, include_classes=open_set_classes)

    if balance_open_set_eval:
        test_dataset_known, test_dataset_unknown = get_equal_len_datasets(test_dataset_known, test_dataset_unknown)

    # Either split train into train and val or use test set as val
    train_dataset = train_dataset_split if split_train_val else train_dataset_whole
    val_dataset = val_dataset_split if split_train_val else test_dataset_known

    if auxiliary_classes is not None:
        all_datasets = {
            'train': train_dataset,
            'val': val_dataset,
            'aux': aux_dataset_known,
            'mix_train': mix_dataset_train,
            'mix_test': mix_dataset_test,
            'test_known': test_dataset_known,
            'test_unknown': test_dataset_unknown,
        }
    else:
        all_datasets = {
            'train': train_dataset,
            'val': val_dataset,
            'test_known': test_dataset_known,
            'test_unknown': test_dataset_unknown,
        }

    return all_datasets

def get_cifar100_datasets(train_transform, test_transform, num_train_classes=80, num_auxiliary_classes=10,
                       num_open_set_classes=10, balance_open_set_eval=False, split_train_val=True, seed=0):

    np.random.seed(seed)
    random_sequence = list(range(100))
    np.random.shuffle(random_sequence)
    train_classes = random_sequence[:num_train_classes]
    auxiliary_classes = random_sequence[num_train_classes:(num_train_classes + num_auxiliary_classes)]
    open_set_classes = random_sequence[(num_train_classes + num_auxiliary_classes):(
            num_train_classes + num_auxiliary_classes + num_open_set_classes)]

    # Init train dataset and subsample training classes
    train_dataset_whole = CustomCIFAR100(root=cifar_100_root, transform=train_transform, train=True)
    train_dataset_whole = subsample_classes(train_dataset_whole, include_classes=train_classes)

    # Split into training and validation sets
    train_dataset_split, val_dataset_split = get_train_val_split(train_dataset_whole)
    val_dataset_split.transform = test_transform

    if auxiliary_classes is not None:
        # aux_dataset_known = CustomCub2011(root=cub_root, transform=train_transform, train=True)
        aux_dataset_whole = CustomCIFAR100(root=cifar_100_root, transform=test_transform, train=False)
        aux_dataset_known = subsample_classes(aux_dataset_whole, include_classes=auxiliary_classes)
        train_dataset_whole3 = CustomCIFAR100(root=cifar_100_root, transform=train_transform, train=True)
        mix_dataset_train = subsample_classes(train_dataset_whole3, include_classes=train_classes+auxiliary_classes)
        train_dataset_whole4 = CustomCIFAR100(root=cifar_100_root, transform=test_transform, train=False)
        mix_dataset_test = subsample_classes(train_dataset_whole4, include_classes=train_classes+auxiliary_classes)

    # Get test set for known classes
    test_dataset_known = CustomCIFAR100(root=cifar_100_root, transform=test_transform, train=False)
    test_dataset_known = subsample_classes(test_dataset_known, include_classes=train_classes)

    # Get testset for unknown classes
    test_dataset_unknown = CustomCIFAR100(root=cifar_100_root, transform=test_transform, train=False)
    test_dataset_unknown = subsample_classes(test_dataset_unknown, include_classes=open_set_classes)

    if balance_open_set_eval:
        test_dataset_known, test_dataset_unknown = get_equal_len_datasets(test_dataset_known, test_dataset_unknown)

    # Either split train into train and val or use test set as val
    train_dataset = train_dataset_split if split_train_val else train_dataset_whole
    val_dataset = val_dataset_split if split_train_val else test_dataset_known

    if auxiliary_classes is not None:
        all_datasets = {
            'train': train_dataset,
            'val': val_dataset,
            'aux': aux_dataset_known,
            'mix_train': mix_dataset_train,
            'mix_test': mix_dataset_test,
            'test_known': test_dataset_known,
            'test_unknown': test_dataset_unknown,
        }
    else:
        all_datasets = {
            'train': train_dataset,
            'val': val_dataset,
            'test_known': test_dataset_known,
            'test_unknown': test_dataset_unknown,
        }

    return all_datasets

if __name__ == '__main__':

    # x = get_cifar_10_100_datasets(None, None, balance_open_set_eval=True)
    x = get_cifar_10_100_datasets(None, None, split_train_val=False, balance_open_set_eval=False)

    print([len(v) for k, v in x.items()])

    debug = 0
