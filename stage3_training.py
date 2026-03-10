import argparse
import math
import os
import random
import shutil
import time
import warnings
from datetime import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import timm

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1, bby1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
    bbx2, bby2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix(images, args):
    lam = np.random.beta(args.cutmix, args.cutmix)
    rand_index = torch.randperm(images.size()[0]).cuda(args.gpu, non_blocking=True)
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
    return images

def mixup(images, args):
    lam = np.random.beta(args.mixup, args.mixup)
    rand_index = torch.randperm(images.size()[0]).cuda(args.gpu, non_blocking=True)
    mixed_images = lam * images + (1 - lam) * images[rand_index]
    return mixed_images


def mix_aug(images, args):
    if args.mix_type == "mixup":
        return mixup(images, args)
    if args.mix_type == "cutmix":
        return cutmix(images, args)
    return images

def get_training_config(args):
    dataset_configs = {
        'CIFAR10': {
            10: (0.1, 16), 50: (0.2, 16), 100: (0.2, 32),
            250: (0.4, 32), 500: (0.6, 64), 1000: (0.9, 128), 1500: (1.0, 128)
        },
        'CIFAR100': {
            10: (0.2, 32), 25: (0.4, 32), 50: (0.6, 64),
            100: (0.9, 128), 150: (1.0, 128)
        },
        'TinyImageNet': {
            10: (0.2, 32), 50: (0.6, 64), 100: (0.9, 128)
        },
        'ImageNet': {
            10: (0.2, 128), 50: (0.2, 256), 100: (0.2, 512)
        },
    }
    configs = dataset_configs.get(args.dataset, {})
    if args.ipc in configs:
        args.min_crop, args.batch_size = configs[args.ipc]
    else:
        print(f"[Warning] IPC {args.ipc} not explicitly configured for {args.dataset}. Using defaults.")
    # ViT architectures on ImageNet use min_crop=1.0
    if args.dataset == 'ImageNet' and args.student_arch in ['vit_b_16', 'vit_s_16']:
        args.min_crop = 1.0

def load_and_prepare_model(arch, num_classes, resume_path=None, gpu=None, dataset='CIFAR10'):
    # ImageNet official pretrained teacher (--teacher-resume official)
    if dataset == 'ImageNet' and resume_path == 'official':
        print(f"=> Loading official pretrained teacher model: {arch}")
        if arch == 'vit_s_16':
            model = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=num_classes)
            return model
        model = models.__dict__[arch](weights='IMAGENET1K_V1')
        return model

    # ViT architecture (non-official)
    if arch == 'vit_s_16':
        model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=num_classes)
        if resume_path:
            if not os.path.isfile(resume_path):
                raise FileNotFoundError(f"=> [ERROR] no checkpoint found at '{resume_path}'")
            print(f"=> loading checkpoint '{resume_path}'")
            loc = f"cuda:{gpu}" if gpu is not None else "cpu"
            checkpoint = torch.load(resume_path, map_location=loc)
            state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict, strict=True)
            print(f"=> loaded checkpoint from {resume_path}. {arch}")
        return model

    # Standard ResNet models
    model = models.__dict__[arch](weights=None)
    if dataset in ['CIFAR10', 'CIFAR100', 'TinyImageNet']:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if resume_path:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"=> [ERROR] no checkpoint found at '{resume_path}'")

        print(f"=> loading checkpoint '{resume_path}'")
        loc = f"cuda:{gpu}" if gpu is not None else "cpu"
        checkpoint = torch.load(resume_path, map_location=loc)

        state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=True)
        print(f"=> loaded checkpoint from {resume_path}. {arch}")

    return model

def build_dataloader(args):
    train_transform = transforms.Compose([                    
        transforms.RandomResizedCrop(size=(args.input_size, args.input_size),
                                     scale=(args.min_crop, 1.0), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(args.input_size, antialias=True),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    train_dataset = datasets.ImageFolder(args.distilled_data_path, transform=train_transform)

    if args.dataset == 'CIFAR10':
        from datasets import cifar10
        val_dataset = cifar10(root=args.real_data_path, train=False, download=True, transform=val_transform)
    elif args.dataset == 'CIFAR100':
        from datasets import cifar100
        val_dataset = cifar100(root=args.real_data_path, train=False, download=True, transform=val_transform)
    elif args.dataset == 'TinyImageNet':
        from datasets import tinyimagenet
        val_dataset = tinyimagenet(root=args.real_data_path, train=False, download=True, transform=val_transform)
    elif args.dataset == 'ImageNet':
        from datasets import imagenet
        val_transform = transforms.Compose([
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        val_dataset = imagenet(root=args.real_data_path, split='val', transform=val_transform)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    print(f"Loaded distilled training set from: {args.distilled_data_path}")
    print(f"Loaded real validation set from: {args.real_data_path}")

    return train_loader, val_loader

def log_summary_to_file(summary_file_path, args, all_accuracies):
    mean_acc = np.mean(all_accuracies)
    std_acc = np.std(all_accuracies)
    var_acc = np.var(all_accuracies)
    write_header = not os.path.exists(summary_file_path)
    with open(summary_file_path, 'a') as f:
        if write_header:
            f.write("Timestamp,Mean_Acc,Std_Dev,Variance,Num_Runs,Temperature,Epochs,LR,Teacher,Student,Dataset_Filename,All_Accuracies\n")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dataset_filename = os.path.basename(args.distilled_data_path)
        accuracies_str = ','.join([f"{acc:.2f}" for acc in all_accuracies])
        log_entry = (
            f"{timestamp},{mean_acc:.2f},{std_acc:.2f},{var_acc:.2f},{args.num_runs},{args.temperature},"
            f"{args.epochs},{args.lr},{args.teacher_arch},{args.student_arch},"
            f"{dataset_filename},\"[{accuracies_str}]\"\n"
        )
        f.write(log_entry)
    print(f"Summary logged to {summary_file_path}")

def train_one_epoch(epoch, train_loader, student_model, optimizer, args, teacher_model=None):
    loss_function_kl = nn.KLDivLoss(reduction="batchmean")
    teacher_model.eval()
    student_model.train()

    t1 = time.time()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.cuda(args.gpu, non_blocking=True)
        labels = labels.cuda(args.gpu, non_blocking=True)
        
        optimizer.zero_grad()

        with torch.no_grad():

            mix_images = mix_aug(images, args)

            soft_mix_label = teacher_model(mix_images)

            soft_mix_label = F.softmax(soft_mix_label / args.temperature, dim=1)

        pred_mix_label = student_model(mix_images)
        soft_pred_mix_label = F.log_softmax(pred_mix_label / args.temperature, dim=1)

        loss = loss_function_kl(soft_pred_mix_label, soft_mix_label)* (args.temperature ** 2)

        loss.backward()
        optimizer.step()

def validate(model, val_loader, args, epoch=None):
    top1 = AverageMeter()
    model.eval()
    with torch.no_grad():
        for images, _, labels in val_loader:
            images = images.cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)
            output = model(images)
            prec1, _ = accuracy(output, labels, topk=(1, 5))
            top1.update(prec1.item(), images.size(0))
    return top1.avg


def main_worker(args,run_id):
    best_acc1 = 0
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    teacher_model = load_and_prepare_model(args.teacher_arch, args.num_classes, args.teacher_resume, args.gpu, args.dataset)
    teacher_model = teacher_model.cuda(args.gpu)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    student_model = load_and_prepare_model(args.student_arch, args.num_classes, gpu=args.gpu, dataset=args.dataset)
    student_model = student_model.cuda(args.gpu)
    student_model.train()

    cudnn.benchmark = True

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            student_model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            student_model.parameters(),
            lr=args.lr,  
            betas=[0.9, 0.999],
            weight_decay=args.wd, 
        )
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD and AdamW are supported.")

    scheduler = LambdaLR(
        optimizer,
        lambda step: 0.5 * (1.0 + math.cos(math.pi * step / args.epochs / 2))
        if step <= args.epochs
        else 0,
        last_epoch=-1,
    )

    train_loader,val_loader = build_dataloader(args)
    
    print(f"Start training run {run_id}...")
    start_time = time.time()
    pbar = tqdm(range(args.epochs), ncols=100)

    for epoch in pbar:
        train_one_epoch(epoch, train_loader, student_model, optimizer, args ,teacher_model=teacher_model)

        if epoch >= (args.epochs-21):  
            acc1 = validate(student_model, val_loader, args, epoch)
            best_acc1 = max(acc1, best_acc1)
            pbar.set_description(f"Epoch[{epoch}] Test Acc: {acc1:.2f}% Best Acc: {best_acc1:.2f}%")

        scheduler.step()

    print(f"\n--- Run {run_id} Complete ---")
    print(f"Best Top-1 Accuracy for this run: {best_acc1:.2f}%")
    total_time = (time.time() - start_time) / 60
    print(f"Training time {total_time:.2f} min")

    return best_acc1

def main():
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.temperature is None:
        if args.mix_type=='mixup': args.temperature=4.0
        elif args.mix_type=="cutmix": args.temperature=20.0
        else: args.temperature=1.0

    dataset_info = {
        'CIFAR10': (10, 32),
        'CIFAR100': (100, 32),
        'TinyImageNet': (200, 64),
        'ImageNet': (1000, 224),
    }
    if args.dataset not in dataset_info:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    args.num_classes, args.input_size = dataset_info[args.dataset]

    get_training_config(args)

    all_accuracies = []

    for i in range(args.num_runs):
        best_acc_run = main_worker(args,i+1)
        all_accuracies.append(best_acc_run)

    print(f"\n{'=' * 20} All Runs Complete {'=' * 20}")
    print(f"Accuracies: {[f'{acc:.2f}%' for acc in all_accuracies]}")
    if len(all_accuracies) > 1:
        print(f"Mean: {np.mean(all_accuracies):.2f}% | Std: {np.std(all_accuracies):.2f}")

    summary_file_path = os.path.join(args.output_dir, "training_summary.csv")
    log_summary_to_file(summary_file_path, args, all_accuracies)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Stage3 Training')
    parser.add_argument('--distilled-data-path', type=str, required=True, help='Path to distilled dataset')
    parser.add_argument('--real-data-path', type=str, required=True, help='Path to original dataset')
    parser.add_argument('--output-dir', type=str, default='./training_output', help='Output directory')
    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=["CIFAR10", "CIFAR100", "TinyImageNet", "ImageNet"], help="dataset")
    parser.add_argument('--input_size', default=32, type=int, help='Image size')
    parser.add_argument('--teacher-arch', type=str, default='resnet18', help='Teacher architecture')
    parser.add_argument('--teacher-resume', type=str, required=True, help='Path to teacher checkpoint')
    parser.add_argument('--student-arch', default='resnet18', help='Student architecture')
    parser.add_argument('--num-classes', default=10, type=int, help='Number of classes')
    parser.add_argument('--epochs', default=500, type=int, help='Total epochs')
    parser.add_argument('--batch-size', default=128, type=int, help='Batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument("--opt", default="adamw", type=str, help="Optimizer (sgd or adamw)")
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--wd', default=0.01, type=float, help='Weight decay')
    parser.add_argument('--mix-type', default='cutmix', type=str, choices=['mixup', 'cutmix'], help='Augmentation type')
    parser.add_argument("--mixup", type=float, default=0.8, help="mixup alpha, mixup enabled if > 0. (default: 0.8)")
    parser.add_argument("--cutmix", type=float, default=1.0, help="cutmix alpha, cutmix enabled if > 0. (default: 1.0)")
    parser.add_argument('--temperature', default=20.0, type=float, help='Temperature for KD. If None, auto-set based on mix-type.')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--seed', default=None, type=int, help='Seed for initializing training.')
    parser.add_argument('--num-runs', type=int, default=3, help='Number of times to run the training and evaluation.')
    parser.add_argument("--ipc", default=10, type=int, help="Images per class")
    parser.add_argument('--min_crop', type=float, default=0.2, help='Minimum crop size for data augmentation')

    main()
