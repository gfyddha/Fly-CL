import random
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset


class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.targets[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
    

def build_transform(is_cifar: bool = False, data_augmentation = None) -> transforms.Compose:
    """ Build a transformation pipeline for image preprocessing. """
    input_size = 224
    resize_im = input_size > 32
    transform = []
    if resize_im:
        size = int((256 / 224) * input_size) if not is_cifar else input_size
        transform.append(transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC))
        transform.append(transforms.CenterCrop(input_size))
    transform.append(transforms.ToTensor())
    if data_augmentation is None:
        pass
    elif data_augmentation == "resnet":
        transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    elif data_augmentation == "vit":
        transform.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    else:
        raise ValueError(f"Unsupported data augmentation: {data_augmentation}")
    return transform


def load_dataset(args, domain_name=None, train=None):
    """ Load a dataset and split it into tasks for continual learning. """
    dataset = args.dataset
    root = args.root
    num_classes = args.num_classes
    num_tasks = args.num_tasks
    batch_size = args.batch_size
    data_augmentation = args.data_augmentation

    # Build transformations
    is_cifar = dataset == "CIFAR-100"
    train_transform = build_transform(is_cifar=is_cifar, data_augmentation=data_augmentation)
    test_transform = build_transform(is_cifar=is_cifar, data_augmentation=data_augmentation)
    train_transform = transforms.Compose([*train_transform])
    test_transform = transforms.Compose([*test_transform])

    # Load the full dataset
    if dataset == "CIFAR-100":
        full_train_dataset = datasets.CIFAR100(root=root, train=True, download=True, transform=train_transform)
        full_test_dataset = datasets.CIFAR100(root=root, train=False, download=True, transform=test_transform)
    elif dataset == "CUB-200-2011":
        full_train_dataset = datasets.ImageFolder(root=f"{root}/cub/train/", transform=train_transform)
        full_test_dataset = datasets.ImageFolder(root=f"{root}/cub/test/", transform=test_transform)
    elif dataset == "VTAB":
        full_train_dataset = datasets.ImageFolder(root=f"{root}/vtab/train/", transform=train_transform)
        full_test_dataset = datasets.ImageFolder(root=f"{root}/vtab/test/", transform=test_transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # Split dataset into tasks
    class_per_task = num_classes // num_tasks
    random_classes = random.sample(list(range(num_classes)), num_classes)
    task_classes = [
        random_classes[i * class_per_task:(i + 1) * class_per_task] 
        for i in range(num_tasks)
    ]

    # Create DataLoader for each task
    train_loader = {}
    test_loader = {}
    for i, classes_in_task in enumerate(task_classes):
        train_subset = Subset(
            full_train_dataset, 
            indices=[index for index, label in enumerate(full_train_dataset.targets) if label in classes_in_task]
        )
        test_subset = Subset(
            full_test_dataset, 
            indices=[index for index, label in enumerate(full_test_dataset.targets) if label in classes_in_task]
        )

        train_loader[i] = DataLoader(train_subset, batch_size=batch_size, shuffle=True, 
                                     num_workers=8, pin_memory=True)
        test_loader[i] = DataLoader(test_subset, batch_size=batch_size, shuffle=False, 
                                    num_workers=8, pin_memory=True)
        # train_loader[i] = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        # test_loader[i] = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
