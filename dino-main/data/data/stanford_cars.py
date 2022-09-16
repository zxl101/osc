import os
import numpy as np
from copy import deepcopy
from scipy import io as mat_io

from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
import torch.utils.data as data
import torchvision.transforms as transforms


# from config import car_root, meta_default_path

car_root = '/home/neuron/Datasets/stanford_car/cars_{}/'
meta_default_path = '/home/neuron/Datasets/stanford_car/car_devkit/devkit/cars_{}.mat'
train_labels = '/home/neuron/Datasets/stanford_car/car_devkit/devkit/train_perfect_preds.txt'
class CarsDataset(Dataset):
    """
        Cars Dataset
    """
    def __init__(self, train=True, limit=0, data_dir=car_root, transform=None, metas=meta_default_path):

        data_dir = data_dir.format('train') if train else data_dir.format('test')
        metas = metas.format('train_annos') if train else metas.format('test_annos_withlabels')
        # metas = metas.format('train_annos') if train else metas.format('test_annos')

        self.loader = default_loader
        self.data_dir = data_dir
        self.data = []
        self.target = []
        self.train = train

        self.transform = transform

        if not isinstance(metas, str):
            raise Exception("Train metas must be string location !")
        labels_meta = mat_io.loadmat(metas)
        # if train:
        #     with open(train_labels, 'r') as file:
        #         lines = file.read().split()
        # print(self.train)
        for idx, img_ in enumerate(labels_meta['annotations'][0]):
            if limit:
                if idx > limit:
                    break

            # self.data.append(img_resized)
            # if len(img_) == 6:
            # if train:
            self.data.append(data_dir + img_[5][0])
                # if self.mode == 'train':
            # if train:
            # self.target.append(lines[idx])
            # else:
            self.target.append(img_[4][0][0])
            # else:
            #     print(img_)

        self.uq_idxs = np.array(range(len(self)))
        self.target_transform = None

    def __getitem__(self, idx):

        image = self.loader(self.data[idx])
        target = self.target[idx] - 1

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        idx = self.uq_idxs[idx]

        return image, target
            # , idx

    def __len__(self):
        return len(self.data)


def subsample_dataset(dataset, idxs):

    dataset.data = np.array(dataset.data)[idxs].tolist()
    dataset.target = np.array(dataset.target)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def subsample_classes(dataset, include_classes=range(160)):

    include_classes_cars = np.array(include_classes) + 1     # SCars classes are indexed 1 --> 196 instead of 0 --> 195
    cls_idxs = [x for x, t in enumerate(dataset.target) if t in include_classes_cars]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_split(train_dataset, val_split=0.2):

    val_dataset = deepcopy(train_dataset)
    train_dataset = deepcopy(train_dataset)

    train_classes = np.unique(train_dataset.target)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.target == cls)[0]

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


def get_scar_datasets(train_transform, test_transform, num_train_classes=98, num_auxiliary_classes=49,
                       num_open_set_classes=49, balance_open_set_eval=False, split_train_val=True, seed=0):
    np.random.seed(seed)
    random_sequence = list(range(196))
    np.random.shuffle(random_sequence)
    train_classes = random_sequence[:num_train_classes]
    auxiliary_classes = random_sequence[num_train_classes:(num_train_classes + num_auxiliary_classes)]
    open_set_classes = random_sequence[(num_train_classes + num_auxiliary_classes):(
                num_train_classes + num_auxiliary_classes + num_open_set_classes)]

    # Init train dataset and subsample training classes
    train_dataset_whole = CarsDataset(data_dir=car_root, transform=train_transform, metas=meta_default_path, train=True)
    train_dataset_whole = subsample_classes(train_dataset_whole, include_classes=train_classes)

    # Split into training and validation sets
    train_dataset_split, val_dataset_split = get_train_val_split(train_dataset_whole)
    val_dataset_split.transform = test_transform

    # Get auxiliary set
    if auxiliary_classes is not None:
        # aux_dataset_known = CustomCub2011(root=cub_root, transform=train_transform, train=True)
        aux_dataset_whole = CarsDataset(data_dir=car_root, transform=train_transform, metas=meta_default_path,
                                        train=True)
        aux_dataset_train = subsample_classes(aux_dataset_whole, include_classes=auxiliary_classes)
        aux_dataset_whole2 = CarsDataset(data_dir=car_root, transform=test_transform, metas=meta_default_path, train=False)
        aux_dataset_test = subsample_classes(aux_dataset_whole2, include_classes=auxiliary_classes)
        train_dataset_whole3 = CarsDataset(data_dir=car_root, transform=train_transform, metas=meta_default_path, train=True)
        mix_dataset_train = subsample_classes(train_dataset_whole3, include_classes=train_classes+auxiliary_classes)
        train_dataset_whole4 = CarsDataset(data_dir=car_root, transform=test_transform, metas=meta_default_path, train=False)
        mix_dataset_test = subsample_classes(train_dataset_whole4, include_classes=train_classes+auxiliary_classes)

    # Get test set for known classes
    test_dataset_known = CarsDataset(data_dir=car_root, transform=test_transform, metas=meta_default_path, train=False)
    test_dataset_known = subsample_classes(test_dataset_known, include_classes=train_classes)

    # Get testset for unknown classes
    test_dataset_unknown = CarsDataset(data_dir=car_root, transform=test_transform, metas=meta_default_path, train=False)
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
            'aux_train': aux_dataset_train,
            'aux_test': aux_dataset_test,
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

def get_scar_loaders(train_classes=range(98), auxiliary_classes=range(98,147),
                       open_set_classes=range(147, 196)):
    train_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                         transforms.RandomCrop((448, 448)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                         transforms.CenterCrop((448, 448)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    datasets = get_scars_datasets(train_transform=train_transform,test_transform=test_transform,train_classes=train_classes,auxiliary_classes=auxiliary_classes,open_set_classes=open_set_classes)
    train_set = datasets['train']
    test_set = datasets['test_known']
    test_unknown_set = datasets['test_unknown']
    train_sampler = RandomSampler(train_set)
    test_sampler = SequentialSampler(test_set)
    test_unknown_sampler = SequentialSampler(test_unknown_set)
    train_loader = DataLoader(train_set,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(test_set,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None
    test_unknown_loader = DataLoader(test_unknown_set,
                             sampler=test_unknown_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None
    return train_loader, test_loader, test_unknown_loader

if __name__ == '__main__':

    x = get_scars_datasets(None, None, split_train_val=False)

    print([len(v) for k, v in x.items()])
    z = [np.unique(v.target) for k, v in x.items()]
    print(z[0])
    print(z[1])
    print(z[2])
    print(z[3])