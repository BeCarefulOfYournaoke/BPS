import argparse
import json
import math
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
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import shutil
from PIL import Image, ImageFile
from tqdm import tqdm

import moco.builder
from datasets import loader
from utils.engine_pretrain import attain_embedding, attain_info_labels, merge_small_clusters, select_hybrid_gmm_centroid_subset
from utils import misc

ImageFile.LOAD_TRUNCATED_IMAGES = True

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

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely " "disable data parallelism."
        )

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

        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    (train_dataset, train_sampler, train_loader, val_dataset, val_loader, test_dataset, test_loader) = build_dataloaders(args)

    print("=> creating model '{}'".format(args.arch))
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
    model_without_ddp = model

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node) 
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
        model_without_ddp = model.module
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

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
            raise FileNotFoundError("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if misc.is_main_process():
        save_path = args.exp_dir
        save_path = os.path.join(save_path, "info_cluster")
        os.makedirs(save_path, exist_ok=True)

        embed_path = os.path.join(save_path, "embedding.npy")
        label_set_path = os.path.join(save_path, "label_set.npy")
        if os.path.exists(embed_path) and os.path.exists(label_set_path):
            embedding_np = np.load(embed_path)
            label_set_np = np.load(label_set_path)
            embedding=torch.from_numpy(embedding_np).cuda()
            label_set=torch.from_numpy(label_set_np).cuda()
            if label_set.dim() > 1:
                label_set = label_set.squeeze()
        else:
            embedding, label_set = attain_embedding(
                    model_without_ddp.encoder_q, val_loader, args
            )
            embedding_np = embedding.cpu().numpy()
            np.save(embed_path, embedding_np)
            label_set_np = label_set.cpu().numpy()
            np.save(label_set_path, label_set_np)
            if label_set.dim() > 1:
                label_set = label_set.squeeze()

        all_images = train_dataset.data
        true_coarse_labels = np.array(train_dataset.targets)
        coarse_class_names = train_dataset.classes
        features_np = embedding.cpu().numpy()

        num_samples, embed_dim = embedding.shape
        info_labels_path = os.path.join(save_path, "info_label.npy")
        if os.path.exists(info_labels_path):
            info_labels = np.load(info_labels_path)
        else:
            info_labels = attain_info_labels(embedding, label_set, args)
            np.save(info_labels_path, info_labels)

        print(f"Loaded {len(info_labels)} info_labels. Min cluster ID: {np.min(info_labels)}, Max cluster ID: {np.max(info_labels)}")

        if args.merge_clusters:
            print("\n--- Pre-processing: Merging small clusters ---")
            merged_info_labels = info_labels.copy()
            unique_coarse_id_for_merge = np.unique(true_coarse_labels)

            for coarse_id in tqdm(unique_coarse_id_for_merge, desc="Merging clusters per class"):
                indices_in_coarse_for_merge = np.where(true_coarse_labels==coarse_id)[0]
                class_info_labels = info_labels[indices_in_coarse_for_merge]
                class_feature = features_np[indices_in_coarse_for_merge]
                new_class_info_label = merge_small_clusters(class_info_labels, class_feature, args.merge_threshold)
                merged_info_labels[indices_in_coarse_for_merge]=new_class_info_label

            info_labels = merged_info_labels
            unique_info_labels,count = np.unique(info_labels,return_counts=True)
            print(f"--- Cluster merging complete | Cluster count = {len(unique_info_labels)} ---")

        all_entropies=None
        if not os.path.exists(args.entropies_path):
            raise FileNotFoundError(f"Entropies file not found at: {args.entropies_path}")
        print(f"--- Loading pre-computed entropies from Entropies file at: {args.entropies_path}")
    
        all_entropies=np.load(args.entropies_path)
        
        if len(all_entropies)!=len(train_dataset):
            raise ValueError(f"Mismatch in number of samples! Dataset has {len(train_dataset)}, but entropies file has {len(all_entropies)}.")

        ipc = args.ipc

        distilled_data_root = os.path.join(args.exp_dir,f'distilled_dataset_ipc{ipc}_k{args.gmm_uncertainty_percentile}_hybrid_ratio{args.hybrid_gmm_centroid_ratio}')

        if os.path.exists(distilled_data_root):
            print(f"Distilled dataset directory already exists, skipping: {distilled_data_root}")
            if args.distributed:
                dist.barrier()
            return

        os.makedirs(distilled_data_root, exist_ok=True)
        print(f"Preparing to distill dataset with ipc={ipc} per class.")

        if len(all_images) != len(embedding):
            print(f"[ERROR] Mismatch! Samples in dataset: {len(all_images)}, Samples in embedding: {len(embedding)}")
            if args.distributed:
                dist.barrier()
            return

        prototype_indices = []
        unique_coarse_id = np.unique(true_coarse_labels)

        for coarse_id in tqdm(unique_coarse_id, desc="Processing Main Classes"):
            indices_in_coarse = np.where(true_coarse_labels == coarse_id)[0]
            if len(indices_in_coarse) <= ipc:
                prototype_indices.extend(indices_in_coarse)
                continue

            sub_concept_labels = info_labels[indices_in_coarse]
            unique_sub_concepts, counts = np.unique(sub_concept_labels, return_counts=True)

            proportions = counts / len(indices_in_coarse)
            ideal_budgets = proportions * ipc
            budgets = np.round(ideal_budgets).astype(int)
            diff = ipc - budgets.sum()
            if diff != 0:
                sorted_indices = np.argsort(counts)[::-1]
                for i in range(abs(diff)):
                    budgets[sorted_indices[i % len(sorted_indices)]] += np.sign(diff)
            budgets[budgets < 0] = 0
          
            for sc_id, budget in zip(unique_sub_concepts, budgets):
                if budget == 0:
                    continue

                mask = (sub_concept_labels == sc_id)
                global_indices_in_sc = indices_in_coarse[mask]

                if len(global_indices_in_sc) <= budget:
                    prototype_indices.extend(global_indices_in_sc)
                    continue
                    
                features_in_sc = features_np[global_indices_in_sc]
                entropies_in_cluster = all_entropies[global_indices_in_sc]

                selected_local_indices_in_cluster = select_hybrid_gmm_centroid_subset(
                    features_in_sc, entropies_in_cluster, budget, args
                )
                selected_indices = global_indices_in_sc[selected_local_indices_in_cluster]
                prototype_indices.extend(selected_indices)

        prototype_indices = sorted(list(set(prototype_indices)))
        print(f"\nExtracted {len(prototype_indices)} unique prototype images from all sub-concepts.")

        print("\nBuilding and saving the distilled dataset...")
        distilled_info_file = open(os.path.join(distilled_data_root, "info.txt"), "w")
        for proto_idx in tqdm(prototype_indices, desc="Saving Distilled Images"):
            img = all_images[proto_idx]
            coarse_label_idx = true_coarse_labels[proto_idx]
            class_name = coarse_class_names[coarse_label_idx].replace(' ', '_')
            coarse_label_dir = os.path.join(distilled_data_root, f"{coarse_label_idx:03d}_{class_name}")
            os.makedirs(coarse_label_dir, exist_ok=True)
            filename = f"cluster{info_labels[proto_idx]}_idx{proto_idx}.png"
            dst_path = os.path.join(coarse_label_dir, filename)
            relative_path = os.path.join(f"{coarse_label_idx:03d}_{class_name}", filename)   
            if args.dataset in ['TinyImageNet', 'ImageNet']:
                shutil.copy(img, dst_path)
            else:
                img_pil = Image.fromarray(img)
                img_pil.save(dst_path)
            distilled_info_file.write(f"{relative_path} {coarse_label_idx}\n")
        distilled_info_file.close()

        print(f"\n--- Distillation complete! ---")
        print(f"Distilled dataset with {len(prototype_indices)} images saved in: {distilled_data_root}")

    if args.distributed:
        dist.barrier()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Stage2 Selection")
    parser.add_argument("--data_dir", type=str, help="path to dataset")
    parser.add_argument("--min_sim", type=float, default=0.5, help="min similarity")
    parser.add_argument("--k_nbr", type=int, default=30, help="num in faiss")
    parser.add_argument("--exp_dir", default="./temp_experiment/")
    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=["CIFAR10", "CIFAR100", "TinyImageNet", "ImageNet"], help="dataset")
    parser.add_argument("--img_size", default=224, type=int, metavar="N", help="img size")
    parser.add_argument("-a", "--arch", metavar="ARCH", default="resnet18", help="model architecture")
    parser.add_argument("-j", "--workers", default=8, type=int, metavar="N", help="workers")
    parser.add_argument("--epochs", default=200, type=int, metavar="N", help="epochs")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("-b", "--batch-size", default=256, type=int, metavar="N", help="batch size")
    parser.add_argument("--lr", default=0.02, type=float, metavar="LR", help="learning rate")
    parser.add_argument("--schedule", default=[120, 160], nargs="*", type=int, help="lr schedule")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, metavar="W", help="weight decay", dest="weight_decay")
    parser.add_argument("--resume", default="", type=str, metavar="PATH", help="checkpoint path")
    parser.add_argument("--world-size", default=-1, type=int, help="nodes number")
    parser.add_argument("--rank", default=-1, type=int, help="node rank")
    parser.add_argument("--dist-url", default="tcp://localhost:23456", type=str, help="dist url")
    parser.add_argument("--dist-backend", default="nccl", type=str, help="dist backend")
    parser.add_argument("--seed", default=1228, type=int, help="seed")
    parser.add_argument("--gpu", default=None, type=int, help="GPU id")
    parser.add_argument("--multiprocessing-distributed", action="store_true", help="multiprocessing distributed")
    parser.add_argument("--moco-dim", default=128, type=int, help="feature dimension")
    parser.add_argument("--moco-k", default=65536, type=int, help="queue size")
    parser.add_argument("--moco-m", default=0.999, type=float, help="moco momentum")
    parser.add_argument("--moco-t", default=0.5, type=float, help="softmax temperature")
    parser.add_argument("--mlp", action="store_true", help="use mlp head")
    parser.add_argument("--ipc", default=10, type=int, help="ipc")
    parser.add_argument('--entropies-path', type=str, default='CIFAR10_train_entropies.npy', help='entropies path')
    parser.add_argument('--gmm-uncertainty-percentile', type=float, default=0.1, help='gmm uncertainty percentile')
    parser.add_argument('--hybrid-gmm-centroid-ratio', type=float, default=0.5, help='hybrid gmm centroid ratio')
    parser.add_argument('--merge-clusters', action='store_true', help='merge clusters')
    parser.add_argument('--merge-threshold', type=int, default=None, help='merge threshold')
    main()
