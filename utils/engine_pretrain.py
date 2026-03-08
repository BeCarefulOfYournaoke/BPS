import torch
import utils.misc as misc
import torch.distributed as dist
import torch.nn as nn
from utils.cluster_opt import Timer
from utils.cluster_opt import get_dist_nbr, cluster_by_infomap
import numpy as np
import time
import sys
from sklearn.mixture import GaussianMixture

def train_one_epoch(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    loss_moco_ = AverageMeter("Loss_moco", ":.4f")
    loss_dino_ = AverageMeter("Loss_dino", ":.4f")
    loss_cls_ = AverageMeter("Loss_cls", ":.4f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, loss_moco_, loss_dino_, loss_cls_, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()

    end = time.time()
    for i, (images, indices, true_labs) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            true_labs = true_labs.cuda(args.gpu, non_blocking=True)

        output, target, dino_loss, cls_loss = model(im_q=images[0], im_k=images[1], coarse_labs=true_labs, args=args, is_train=True)

        loss_moco = criterion(output, target)
        alpha = args.weight_cls
        if (epoch+1)<=args.warmup_epoch:
            loss = alpha*cls_loss + (1-alpha)*loss_moco
        else:
            loss = alpha*cls_loss + (1-alpha)*loss_moco + ( (epoch+1-args.warmup_epoch) / (args.epochs-args.warmup_epoch) * args.weight_info) * dino_loss

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        loss_moco_.update(loss_moco.item(), images[0].size(0))
        loss_dino_.update(dino_loss.item(), images[0].size(0))
        loss_cls_.update(cls_loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            sys.stdout.flush()

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    loss_moco_ = AverageMeter("Loss_moco", ":.4f")
    loss_dino_ = AverageMeter("Loss_dino", ":.4f")
    loss_cls_ = AverageMeter("Loss_cls", ":.4f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, loss_moco_, loss_dino_, loss_cls_, top1, top5],
        prefix="Test: ",
    )

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, _, true_labs) in enumerate(val_loader):
            if args.gpu is not None:
                images[0] = images[0].cuda(args.gpu, non_blocking=True)
                images[1] = images[1].cuda(args.gpu, non_blocking=True)
                true_labs = true_labs.cuda(args.gpu, non_blocking=True)
            output, target, dino_loss, cls_loss = model(im_q=images[0], im_k=images[1], coarse_labs=true_labs, args=args, is_train=False)

            loss_moco = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            alpha = args.weight_cls
            
            loss = alpha * cls_loss + (1 - alpha) * loss_moco  

            losses.update(loss.item(), images[0].size(0))
            loss_moco_.update(loss_moco.item(), images[0].size(0))
            loss_dino_.update(dino_loss.item(), images[0].size(0)) 
            loss_cls_.update(cls_loss.item(), images[0].size(0))
            top1.update(acc1[0], images[0].size(0))
            top5.update(acc5[0], images[0].size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


@torch.no_grad()
def attain_embedding(encoder, val_loader, args):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Attain embedding:"

    encoder.eval()
    features_list = []
    targets_list = []

    for i, (images, _, target) in enumerate(metric_logger.log_every(val_loader, 20, header)):
        images = images.cuda(args.gpu, non_blocking=True)
        target_gpu = target.cuda(args.gpu, non_blocking=True)

        output = encoder(images, feat=True)

        features_list.append(output.cpu())
        targets_list.append(target_gpu.cpu()) 

    features = torch.cat(features_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    
    features = torch.nn.functional.normalize(features, dim=1)

    print(f"val_loader size is: {len(val_loader.dataset)}", flush=True)
    print(f"feature size: {features.size()}, label size: {targets.size()}", flush=True)

    return features, targets.squeeze()


def attain_info_labels(embedding, coarse_lab, args):
    min_sim = args.min_sim
    k_nbr = args.k_nbr
    print(len(coarse_lab))
    info_label = np.zeros([len(coarse_lab)], dtype=np.int64)

    print("****************************************************")
    with Timer("start infomap cluster step with rank 0"):
        base = 0
        info_feature_norm_cpu = torch.nn.functional.normalize(embedding, dim=1).cpu().numpy()
        for c_lab in torch.unique(coarse_lab):
            mask = (coarse_lab == c_lab)
            mask_cpu = mask.cpu().numpy()
           
            info_feature = info_feature_norm_cpu[mask_cpu]

            dists, nbrs = get_dist_nbr(feature=info_feature, k_nbr=k_nbr, nproc=1, index_path="", knn_method="faiss-gpu", verbose=True)

            pred_labels, idx2label, class_num, idx_len = cluster_by_infomap(nbrs, dists, min_sim=min_sim)

            info_label[mask_cpu] = (base + pred_labels.astype(np.int64))
            base += len(np.unique(pred_labels))
        print(f"total {base} info fine classes")
    print("****************************************************")
    return info_label

class AverageMeter(object):

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"



def merge_small_clusters(class_info_labels, class_features, k_threshold):
    unique_labels, count = np.unique(class_info_labels, return_counts=True)
    small_cluster_mask = count < k_threshold
    large_cluster_mask = count >= k_threshold
    small_cluster_ids = unique_labels[small_cluster_mask]
    large_cluster_ids = unique_labels[large_cluster_mask]

    if len(small_cluster_ids) == 0 or len(large_cluster_ids) == 0:
        return class_info_labels

    print(f" Found {len(small_cluster_ids)} small clusters to merge (size < {k_threshold})...")
    centroids = {}
    for label in unique_labels:
        centroids[label] = np.mean(class_features[class_info_labels == label], axis=0)

    large_centroids_list = np.array([centroids[l_id] for l_id in large_cluster_ids])
    norm_large_centroids = large_centroids_list / (np.linalg.norm(large_centroids_list, axis=1, keepdims=True) + 1e-8)

    new_labels = class_info_labels.copy()
    for s_id in small_cluster_ids:
        small_centroid = centroids[s_id].reshape(1, -1)
        norm_small_centroids = small_centroid / (np.linalg.norm(small_centroid) + 1e-8)
        similarities = norm_small_centroids @ norm_large_centroids.T
        most_similar_large_cluster_idx = np.argmax(similarities)
        most_similar_large_cluster_id = large_cluster_ids[most_similar_large_cluster_idx]
        new_labels[class_info_labels == s_id] = most_similar_large_cluster_id

    return new_labels


def allocate_budget(total_budget, ratios):
    ideal_budgets = total_budget * np.array(ratios)
    int_budgets = np.floor(ideal_budgets).astype(int)
    residuals = ideal_budgets - int_budgets
    remaining_budget = total_budget - int_budgets.sum()
    sorted_indices = np.argsort(residuals)[::-1]
    for i in range(int(remaining_budget)):
        int_budgets[sorted_indices[i]] += 1
    return int_budgets


def select_gmm_entropy_subset(cluster_entropies, budget, args, cluster_id_for_logging=None):
    n_cluster = len(cluster_entropies)
    if n_cluster <= budget:
        return np.arange(n_cluster)

    selected_indices = []
    gmm_selected_indices = []

    if n_cluster >= 10:
        select_num = int(max(n_cluster * args.gmm_uncertainty_percentile, min(10, n_cluster)))
        sorted_cluster_entropies = cluster_entropies[np.argsort(cluster_entropies)]
        candidate_entropies = sorted_cluster_entropies[-select_num:]

        if len(candidate_entropies) >= 5:
            try:
                gmm = GaussianMixture(n_components=2, random_state=args.seed).fit(candidate_entropies.reshape(-1, 1))
                sorted_indices_gmm = np.argsort(gmm.means_.flatten())
                means = gmm.means_.flatten()[sorted_indices_gmm]
                covs = gmm.covariances_.flatten()[sorted_indices_gmm]
                weights = gmm.weights_[sorted_indices_gmm]

                mu_left, var_left, w_left = means[0], covs[0], weights[0]
                mu_right, var_right, w_right = means[1], covs[1], weights[1]

                a = 1 / (2 * var_right) - 1 / (2 * var_left)
                b = mu_left / var_left - mu_right / var_right
                c = (mu_right ** 2 / (2 * var_right) - mu_left ** 2 / (2 * var_left)) + np.log(w_left * np.sqrt(var_left) / (w_right * np.sqrt(var_right)))
                roots = np.roots([a, b, c])

                intersection_point = np.inf
                valid_roots = roots[(roots > mu_left) & (roots < mu_right)]
                if len(valid_roots) > 0:
                    intersection_point = valid_roots[0]

                lower_bound = mu_left
                upper_bound = intersection_point

                primary_pool_mask = (cluster_entropies >= lower_bound) & (cluster_entropies <= upper_bound)
                primary_pool_local_indices = np.where(primary_pool_mask)[0]
                secondary_pool_mask = cluster_entropies < lower_bound
                secondary_pool_local_indices = np.where(secondary_pool_mask)[0]

                if len(primary_pool_local_indices) >= budget:
                    primary_pool_entropies = cluster_entropies[primary_pool_local_indices]
                    sorted_primary_pool_indices = primary_pool_local_indices[np.argsort(primary_pool_entropies)]
                    gmm_selected_indices = list(sorted_primary_pool_indices[:budget])
                else:
                    gmm_selected_indices = list(primary_pool_local_indices)
                    num_remaining = budget - len(gmm_selected_indices)
                    if num_remaining > 0 and len(secondary_pool_local_indices) > 0:
                        secondary_pool_entropies = cluster_entropies[secondary_pool_local_indices]
                        sorted_secondary_pool_indices = secondary_pool_local_indices[np.argsort(secondary_pool_entropies)[::-1]]
                        num_to_take = min(num_remaining, len(sorted_secondary_pool_indices))
                        gmm_selected_indices.extend(sorted_secondary_pool_indices[:num_to_take])
            except ValueError:
                print("GMM failed !")
                pass

    selected_indices = gmm_selected_indices
    num_current_selected = len(selected_indices)
    num_shortfall = budget - num_current_selected
    
    if cluster_id_for_logging is not None:
        print(f" [Cluster {cluster_id_for_logging}] -> GMM selected {num_current_selected}/{budget}. Backfilling {num_shortfall} samples.")

    if num_shortfall > 0:
        all_local_indices = np.arange(n_cluster)
        available_for_fallback = np.setdiff1d(all_local_indices, selected_indices, assume_unique=True)
        fallback_entropies = cluster_entropies[available_for_fallback]
        sorted_fallback_indices = available_for_fallback[np.argsort(fallback_entropies)[::-1]]
        num_to_take_from_fallback = min(num_shortfall, len(sorted_fallback_indices))
        backfill_indices = sorted_fallback_indices[:num_to_take_from_fallback]
        selected_indices.extend(backfill_indices)

    return selected_indices

def select_hybrid_gmm_centroid_subset(cluster_features, cluster_entropies, budget, args):
    n_cluster = len(cluster_features)
    if n_cluster <= budget:
        return np.arange(n_cluster)

    num_from_centroid = int(budget * args.hybrid_gmm_centroid_ratio + 0.5)
    num_from_gmm = budget - num_from_centroid

    centroid = np.mean(cluster_features, axis=0)
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
    similarities = np.dot(cluster_features, centroid_norm)
    sorted_centroid_indices = np.argsort(similarities)[::-1]
    centroid_selected_indices = sorted_centroid_indices[:num_from_centroid]

    gmm_selected_indices = np.array([], dtype=int)
    if num_from_gmm > 0:
        is_selected_mask = np.zeros(n_cluster, dtype=bool)
        is_selected_mask[centroid_selected_indices] = True
        remaining_local_indices = np.where(~is_selected_mask)[0]
        remaining_entropies = cluster_entropies[remaining_local_indices]

        if len(remaining_local_indices) > 0:
            gmm_selected_local_indices_in_remainder = select_gmm_entropy_subset(remaining_entropies, num_from_gmm, args)
            gmm_selected_indices = remaining_local_indices[gmm_selected_local_indices_in_remainder]

    combined_indices = np.concatenate([centroid_selected_indices, gmm_selected_indices])
    num_shortfall = budget - len(combined_indices)
    if num_shortfall > 0:
        fallback_candidates = [idx for idx in sorted_centroid_indices if idx not in combined_indices]
        num_to_take_from_fallback = min(num_shortfall, len(fallback_candidates))
        combined_indices = np.concatenate([combined_indices, fallback_candidates[:num_to_take_from_fallback]])

    return np.array(list(set(combined_indices)), dtype=int)
