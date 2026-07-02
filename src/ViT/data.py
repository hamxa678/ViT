"""
Data loading and splitting utilities.

Expects a folder structure like:

    data_dir/
        class_a/
            img1.jpg
            img2.jpg
        class_b/
            img1.jpg
            ...

This is the standard torchvision ImageFolder layout.
"""

import logging

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

logger = logging.getLogger("ViT")


def build_transforms(img_size=224, augment=True):
    """Standard ImageNet-style normalization + optional light augmentation."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if augment:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, eval_tf


def prepare_dataloaders(
    data_dir,
    img_size=224,
    batch_size=32,
    train_split=0.7,
    val_split=0.15,
    test_split=0.15,
    num_workers=2,
    augment=True,
    seed=42,
):
    """
    Loads an ImageFolder-style dataset from `data_dir` and splits it into
    train/val/test DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader, class_names (list[str])
    """
    assert abs((train_split + val_split + test_split) - 1.0) < 1e-6, \
        "train_split + val_split + test_split must sum to 1.0"

    train_tf, eval_tf = build_transforms(img_size, augment)

    # Load once with eval transform to determine sizes/classes, then
    # rebuild the train subset with augmentation applied.
    full_dataset = datasets.ImageFolder(root=data_dir, transform=eval_tf)
    class_names = full_dataset.classes
    n = len(full_dataset)

    n_train = int(n * train_split)
    n_val = int(n * val_split)
    n_test = n - n_train - n_val  # remainder to avoid rounding loss

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    # Apply augmentation only to the training subset.
    if augment:
        train_dataset_aug = datasets.ImageFolder(root=data_dir, transform=train_tf)
        train_subset.dataset = train_dataset_aug

    logger.info(
        "Dataset loaded from '%s': %d total images, %d classes -> %s",
        data_dir, n, len(class_names), class_names,
    )
    logger.info(
        "Split -> train: %d | val: %d | test: %d", n_train, n_val, n_test
    )

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_names
