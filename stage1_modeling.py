import argparse
import os
import random
import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torchvision.models as models

import moco.builder
from utils.engine_pretrain import train_one_epoch
from utils import misc
from datasets import loader

torch.cuda.memory.max_split_size_mb = 128


def build_dataloaders(args):
    return getattr(loader, args.dataset)(args)


def main():
    args = parser.parse_args()

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.multiprocessing_distributed:
        misc.setup_for_distributed(args.gpu == 0)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])  
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url, 
            world_size=args.world_size, 
            rank=args.rank,             
        )

    (
        train_dataset,
        train_sampler,
        train_loader,
        val_dataset,
        val_loader,
        test_dataset,
        test_loader,
    ) = build_dataloaders(args)

    model = moco.builder.MoCo(
        models.__dict__[args.arch],
        args.classes_num,
        args.dataset,
        args.moco_dim,
        args.moco_k,
        args.moco_m,
        args.moco_t,
        args.mlp,
    )

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node) 
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint["epoch"]))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        misc.adjust_learning_rate(optimizer, epoch, args)

        train_one_epoch(train_loader, model, criterion, optimizer, epoch, args)

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            if (epoch >= (args.epochs-21)) and (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
                os.makedirs(args.exp_dir, exist_ok=True)
                misc.save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": args.arch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best=False,
                    filename=os.path.join(args.exp_dir, "checkpoint_{:04d}.pth.tar".format(epoch+1)),
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Stage1 Modeling")

    parser.add_argument("--data_dir", type=str, help="path to dataset")
    parser.add_argument("--min_sim", type=float, default=0.5, help="min similarity for construct graph")
    parser.add_argument("--k_nbr", type=int, default=30, help="num in faiss to search nearest k_nbr")
    parser.add_argument("--weight_info", type=float, default=1.0, help="the weight of info loss")
    parser.add_argument("--weight_cls", type=float, default=0.5, help="the weight of cls loss and contrastive loss")
    parser.add_argument("--exp_dir", default="./temp_experiment/")
    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=["CIFAR10", "CIFAR100", "TinyImageNet", "ImageNet"], help="dataset")
    parser.add_argument("--img_size", default=224, type=int, metavar="N", help="width & heigh of img")
    parser.add_argument("-a", "--arch", metavar="ARCH", default="resnet18", help="model architecture")
    parser.add_argument("-j", "--workers", default=8, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument('--warmup_epoch', default=5, type=int, help='warm up epoch')
    parser.add_argument("--epochs", default=200, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="manual epoch number")
    parser.add_argument("-b", "--batch_size", default=256, type=int, metavar="N", help="mini-batch size")
    parser.add_argument("--lr", "--learning-rate", default=0.04, type=float, metavar="LR", help="initial learning rate", dest="lr")
    parser.add_argument("--schedule", default=[120, 160], nargs="*", type=int, help="learning rate schedule")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver")
    parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, metavar="W", help="weight decay", dest="weight_decay")
    parser.add_argument("-p", "--print-freq", default=40, type=int, metavar="N", help="print frequency")
    parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint")
    parser.add_argument("--world-size", default=-1, type=int, help="number of nodes for distributed training")
    parser.add_argument("--rank", default=-1, type=int, help="node rank for distributed training")
    parser.add_argument("--dist-url", default="tcp://localhost:10001", type=str, help="url used to set up distributed training")
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--seed", default=1228, type=int, help="seed for initializing training")
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use")
    parser.add_argument("--multiprocessing-distributed", action="store_true", help="Use multi-processing distributed training")
    parser.add_argument("--moco-dim", default=128, type=int, help="feature dimension")
    parser.add_argument("--moco-k", default=65536, type=int, help="queue size; number of negative keys")
    parser.add_argument("--moco-m", default=0.999, type=float, help="moco momentum of updating key encoder")
    parser.add_argument("--moco-t", default=0.5, type=float, help="softmax temperature")
    parser.add_argument("--mlp", action="store_true", help="use mlp head")
    parser.add_argument("--aug-plus", action="store_true", help="use moco v2 data augmentation")
    parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")

    main()
