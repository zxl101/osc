import os
import pandas as pd
import numpy as np
from copy import deepcopy
# import random

from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler, SequentialSampler
# from config import cub_root


cub_root = '/home/neuron/Datasets/CUB'
class CustomCub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root=cub_root, train=True, transform=None, target_transform=None, loader=default_loader, download=False):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.loader = loader
        self.train = train


        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.uq_idxs = np.array(range(len(self)))

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target\
            # , self.uq_idxs[idx]


def subsample_dataset(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.data = dataset.data[mask]
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def subsample_classes(dataset, include_classes=range(150)):

    include_classes_cub = np.array(include_classes) + 1     # CUB classes are indexed 1 --> 200 instead of 0 --> 199
    cls_idxs = [x for x, (_, r) in enumerate(dataset.data.iterrows()) if int(r['target']) in include_classes_cub]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_split(train_dataset, val_split=0.2):

    val_dataset = deepcopy(train_dataset)
    train_dataset = deepcopy(train_dataset)

    train_classes = np.unique(train_dataset.data['target'])

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.data['target'] == cls)[0]

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

        rand_idxs = np.random.choice(range(len(dataset1)), size=(len(dataset2,)), replace=False)
        subsample_dataset(dataset1, rand_idxs)

    elif len(dataset2) > len(dataset1):

        rand_idxs = np.random.choice(range(len(dataset2)), size=(len(dataset1,)), replace=False)
        subsample_dataset(dataset2, rand_idxs)

    return dataset1, dataset2

def get_cub_datasets(train_transform, test_transform, num_train_classes=100, num_auxiliary_classes=50,
                       num_open_set_classes=50, balance_open_set_eval=False, split_train_val=True, seed=1):

    np.random.seed(seed)
    random_sequence = list(range(200))
    np.random.shuffle(random_sequence)
    train_classes = random_sequence[:num_train_classes]
    auxiliary_classes = random_sequence[num_train_classes:(num_train_classes+num_auxiliary_classes)]
    open_set_classes = random_sequence[(num_train_classes+num_auxiliary_classes):(num_train_classes+num_auxiliary_classes+num_open_set_classes)]

    # Init train dataset and subsample training classes
    train_dataset_whole = CustomCub2011(root=cub_root, transform=train_transform, train=True)
    train_dataset_whole = subsample_classes(train_dataset_whole, include_classes=train_classes)

    # Split into training and validation sets
    train_dataset_split, val_dataset_split = get_train_val_split(train_dataset_whole)
    val_dataset_split.transform = test_transform

    # Get auxiliary set
    if auxiliary_classes is not None:
        # aux_dataset_known = CustomCub2011(root=cub_root, transform=train_transform, train=True)
        aux_dataset_whole = CustomCub2011(root=cub_root, transform=train_transform, train=True)
        aux_dataset_train = subsample_classes(aux_dataset_whole, include_classes=auxiliary_classes)
        aux_dataset_whole2 = CustomCub2011(root=cub_root, transform=test_transform, train=False)
        aux_dataset_test = subsample_classes(aux_dataset_whole2, include_classes=auxiliary_classes)
        train_dataset_whole3 = CustomCub2011(root=cub_root, transform=train_transform, train=True)
        mix_dataset_train = subsample_classes(train_dataset_whole3, include_classes=train_classes+auxiliary_classes)
        train_dataset_whole4 = CustomCub2011(root=cub_root, transform=test_transform, train=False)
        mix_dataset_test = subsample_classes(train_dataset_whole4, include_classes=train_classes+auxiliary_classes)


    # Get test set for known classes
    test_dataset_known = CustomCub2011(root=cub_root, transform=test_transform, train=False)
    test_dataset_known = subsample_classes(test_dataset_known, include_classes=train_classes)

    # Get testset for unknown classes
    test_dataset_unknown = CustomCub2011(root=cub_root, transform=test_transform, train=False)
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
            'mix_train' : mix_dataset_train,
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

def get_cub_loaders(train_classes=range(100), auxiliary_classes=range(100,150),
                       open_set_classes=range(150, 200)):
    train_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                         transforms.RandomCrop((448, 448)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                         transforms.CenterCrop((448, 448)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    datasets = get_cub_datasets(train_transform=train_transform,test_transform=test_transform,train_classes=train_classes,auxiliary_classes=auxiliary_classes,open_set_classes=open_set_classes)
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

    x = get_cub_datasets(None, None, split_train_val=False, train_classes=np.random.choice(range(200), size=100, replace=False))
    print([len(v) for k, v in x.items()])
    # z = x['train'][0]
    # debug = 0
