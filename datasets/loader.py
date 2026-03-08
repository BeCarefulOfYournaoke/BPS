import random
from PIL import ImageFilter
import torch
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.transforms import RandAugment 

from datasets import *

class TwoCropsTransform:

    def __init__(self, base_transform, weak=False, weak_aug=None):
        self.base_transform = base_transform
        self.weak_transform = weak_aug
        self.weak = weak
    def __call__(self, x):
        if not self.weak:
            q = self.base_transform(x)
            k = self.base_transform(x)
        else:
            q = self.weak_transform(x)
            k = self.base_transform(x)
        return [q, k]

class GaussianBlur(object):
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def set_augmentation(args, mode = 'none'):
    fig_size = args.img_size
    min_crop = 0.2

    if args.dataset in ['CIFAR10', 'CIFAR100', 'ImageNet', 'TinyImageNet']:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    else:
        raise ValueError(f"dataset should not be {args.dataset}!")

    if mode == "strong":
        if args.dataset in ['ImageNet']:
            res_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=[fig_size, fig_size], scale=(min_crop, 1)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomPerspective(0.5, 0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.ToTensor(),
                normalize,
            ])     
        else:
            res_transform = transforms.Compose([
                transforms.RandomResizedCrop(fig_size, scale=(min_crop, 1.0)),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

    elif mode == 'weak':
        if args.dataset in ['ImageNet']:
            res_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=[fig_size, fig_size], scale=(min_crop, 1)),
                transforms.RandomPerspective(0.5, 0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            res_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=[fig_size, fig_size], scale=(min_crop, 1)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    else:  
        ratio = 224 / 256
        before_size = int(fig_size/ratio)
        res_transform = transforms.Compose([
            transforms.Resize(before_size),
            transforms.CenterCrop(fig_size),
            transforms.ToTensor(),
            normalize,
        ])
    return res_transform

def CIFAR10(args):
    root = args.data_dir
    strong_augmentation = set_augmentation(args, mode = 'strong')
    test_augmentation = set_augmentation(args, mode = 'test')
    augmentation = TwoCropsTransform(strong_augmentation)
    trainval_dataset = cifar10(root=root, train=True, download=True, transform=augmentation)
    val_dataset = cifar10(root=root, train=True, download=True, transform=test_augmentation)
    test_dataset = cifar10(root=root, train=False, download=True, transform=test_augmentation)

    args.classes_num = trainval_dataset.cls_num
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainval_dataset)
    else:
        train_sampler = None
    
    train_loader = torch.utils.data.DataLoader(
        trainval_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.workers, 
        pin_memory=True, 
        drop_last=False
        )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.workers, 
        pin_memory=True, 
        drop_last=False
        )

    return trainval_dataset, train_sampler, train_loader, val_dataset, val_loader, test_dataset, test_loader


def CIFAR100(args):
    root = args.data_dir
    strong_augmentation =set_augmentation(args, mode = 'strong')
    test_augmentation = set_augmentation(args, mode = 'test')
    augmentation = TwoCropsTransform(strong_augmentation)
    
    trainval_dataset = cifar100(root=root, train=True, download=True, transform=augmentation)

    val_dataset = cifar100(root=root, train=True, download=True, transform=test_augmentation)
    test_dataset = cifar100(root=root, train=False, download=True, transform=test_augmentation)

    args.classes_num = trainval_dataset.cls_num

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainval_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        trainval_dataset,
        batch_size=args.batch_size,
        
        shuffle=(train_sampler is None),
        num_workers=args.workers, 
        pin_memory=True,   
        sampler=train_sampler,  
        drop_last=True,   
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.workers, 
        pin_memory=True, 
        drop_last=False   
        )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.workers, 
        pin_memory=True, 
        drop_last=False
        )

    return trainval_dataset, train_sampler, train_loader, val_dataset, val_loader, test_dataset, test_loader



def TinyImageNet(args):
    root = args.data_dir
    strong_augmentation = set_augmentation(args, mode='strong')
    test_augmentation = set_augmentation(args, mode='test')
    augmentation = TwoCropsTransform(strong_augmentation)

    trainval_dataset = tinyimagenet(root=root, train=True, download=True, transform=augmentation)
    val_dataset = tinyimagenet(root=root, train=True, download=True, transform=test_augmentation)
    test_dataset = tinyimagenet(root=root, train=False, download=True, transform=test_augmentation)
    
    args.classes_num = trainval_dataset.cls_num

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainval_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        trainval_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,  
        pin_memory=True, 
        sampler=train_sampler,  
        drop_last=True,  
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False  
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False
    )

    return trainval_dataset, train_sampler, train_loader, val_dataset, val_loader, test_dataset, test_loader

def ImageNet(args):
    strong_augmentation = set_augmentation(args, mode = 'strong')
    test_augmentation = set_augmentation(args, mode = 'test')
    augmentation = TwoCropsTransform(strong_augmentation)
    
    trainval_dataset = imagenet(root=args.data_dir, split="train", transform=augmentation)
    val_dataset = imagenet(root=args.data_dir, split="train", transform=test_augmentation)
    test_dataset = imagenet(root=args.data_dir, split="val", transform=test_augmentation)

    args.classes_num = trainval_dataset.cls_num

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainval_dataset)
    else:
        train_sampler = None
    
    train_loader = torch.utils.data.DataLoader(
        trainval_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.workers, 
        pin_memory=True, 
        drop_last=False
        )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.workers, 
        pin_memory=True, 
        drop_last=False
        )

    return trainval_dataset, train_sampler, train_loader, val_dataset, val_loader, test_dataset, test_loader