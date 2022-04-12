from avalanche.benchmarks.classic import SplitCUB200, SplitCIFAR100
from avalanche.benchmarks import nc_benchmark
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data.dataset import Dataset
import cv2
import os
from tqdm import tqdm
from imutils import paths
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from scipy.io import loadmat


def get_cifar(resnet):
    config = resolve_data_config({}, model=resnet)
    transform = create_transform(**config)
    split = SplitCIFAR100(n_experiences=20, seed=0, shuffle=True, train_transform=transform,
                          eval_transform=transform)
    return split


def get_cub(resnet):
    config = resolve_data_config({}, model=resnet)
    transform = create_transform(**config)
    split = SplitCUB200(n_experiences=10, classes_first_batch=20, seed=0, shuffle=True, train_transform=transform,
                        eval_transform=transform)
    return split




class CustomDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.data = images
        self.targets = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index][:]

        if self.transforms:
            data = self.transforms(data)

        if self.targets is not None:
            return (data, self.targets[index])
        else:
            return data


def get_caltech(resnet):
    config = resolve_data_config({}, model=resnet)
    transform = create_transform(**config)
    transform = transforms.Compose([transforms.ToPILImage(), transform])
    image_paths = list(paths.list_images('data/caltech101/caltech101/101_ObjectCategories'))

    data = []
    labels = []
    for img_path in tqdm(image_paths):
        label = img_path.split(os.path.sep)[-2]
        if label == "BACKGROUND_Google":
            continue
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        data.append(img)
        labels.append(label)

    data = np.array(data)
    labels = np.array(labels)
    lb = LabelEncoder()
    labels = lb.fit_transform(labels)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, stratify=labels, test_size=0.2, random_state=0)
    train_set = CustomDataset(x_train, y_train, transform)
    test_set = CustomDataset(x_test, y_test, transform)
    split = nc_benchmark(
        train_dataset=train_set,
        test_dataset=test_set,
        n_experiences=11,
        task_labels=False,
        seed=0,
        fixed_class_order=None,
        shuffle=True,
        per_exp_classes={10: 1},
        class_ids_from_zero_in_each_exp=False,
        train_transform=transform,
        eval_transform=transform)
    # split.n_experiences = 10
    return split


def get_flowers(resnet):
    config = resolve_data_config({}, model=resnet)
    transform = create_transform(**config)
    transform = transforms.Compose([transforms.ToPILImage(), transform])
    image_paths = list(paths.list_images('data/flowers102'))

    image_paths = sorted(image_paths)

    labels = loadmat('data/flowers102/imagelabels.mat')['labels'][0]
    setid = loadmat('data/flowers102/setid.mat')
    train_id = setid['trnid'][0] - 1
    test_id = setid['tstid'][0] - 1
    data = []
    for i, img_path in tqdm(enumerate(image_paths)):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        data.append(img)

    data = np.array(data)
    labels = np.array(labels) - 1
    x_train, x_test, y_train, y_test = train_test_split(data, labels, stratify=labels, test_size=0.2, random_state=0)

    train_set = CustomDataset(x_train, y_train, transform)
    test_set = CustomDataset(x_test, y_test, transform)
    split = nc_benchmark(
        train_dataset=train_set,
        test_dataset=test_set,
        n_experiences=11,
        task_labels=False,
        seed=0,
        fixed_class_order=None,
        shuffle=True,
        per_exp_classes={10: 2},
        class_ids_from_zero_in_each_exp=False,
        train_transform=transform,
        eval_transform=transform)
    # split.n_experiences = 10
    return split


def get(net, dataset):
    if dataset == 'cub':
        return get_cub(net)
    elif dataset == 'cifar':
        return get_cifar(net)
    elif dataset == 'flowers':
        return get_flowers(net)
    elif dataset == 'caltech':
        return get_caltech(net)



