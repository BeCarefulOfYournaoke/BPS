import torch
import torch.nn as nn
import torch.distributed as dist

import numpy as np
from utils.cluster_opt import get_dist_nbr, cluster_by_infomap, Timer
from utils import misc
import math

class ModelBase(nn.Module):
    def __init__(self, base_encoder, cls_out_dim, dim=128, pretrained=True):
        super(ModelBase, self).__init__()
        self.net = base_encoder(pretrained=pretrained)
        self.embed_dim = self.net.fc.weight.shape[1]
        self.net.fc = nn.Identity() 
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, cls_out_dim)
        )
        self.projector = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, dim)
        )

    def forward(self, x, feat=False):
        x = self.net(x)
        if feat:
            return x
        else:
            logit = self.classifier(x)
            proj = self.projector(x)
            return logit, proj

class MoCo(nn.Module):
    def __init__(self, base_encoder, cls_out_dim, dataset, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        self.len_queue = K

        if dataset in ["ImageNet"]:
            self.encoder_q = ModelBase(base_encoder, cls_out_dim, dim, False)
            self.encoder_k = ModelBase(base_encoder, cls_out_dim, dim, False)
        else:
            self.encoder_q = ModelBase(base_encoder, cls_out_dim, dim, True)
            self.encoder_k = ModelBase(base_encoder, cls_out_dim, dim, True)

        self.embed_dim = self.encoder_q.embed_dim

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  
            param_k.requires_grad = False  

        self.register_buffer("queue_emb", torch.randn(K, dim))
        self.queue_emb = nn.functional.normalize(self.queue_emb, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_emb_copy", torch.randn(K, dim))
        self.queue_emb_copy = nn.functional.normalize(self.queue_emb_copy, dim=1)

        self.register_buffer("info_label", torch.zeros(K, dtype=torch.int))
        
        self.register_buffer("queue_coarse_lab", torch.zeros(K, dtype=torch.int))
        self.register_buffer("queue_coarse_lab_copy", torch.zeros(K, dtype=torch.int))

        self.dino_loss_criterion = DINOLoss(dim)
        self.cls_loss_criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_emb, keys_coarse_lab):
        keys_emb = concat_all_gather(keys_emb)

        batch_size = keys_emb.shape[0]
        ptr = int(self.queue_ptr)
        assert self.len_queue % batch_size == 0

        self.queue_emb[ptr : ptr + batch_size, :] = keys_emb

        keys_coarse_lab = concat_all_gather(keys_coarse_lab)
        self.queue_coarse_lab[ptr : ptr + batch_size] = keys_coarse_lab

        ptr = (ptr + batch_size) % self.len_queue
        if ptr == 0:
            self.queue_emb_copy.data.copy_(self.queue_emb.data)
            self.queue_coarse_lab_copy.data.copy_(self.queue_coarse_lab.data)
            # self.queue_true_lab_copy.data.copy_(self.queue_true_lab.data)

        self.queue_ptr[0] = ptr

   
    @torch.no_grad()
    def _info_cluster(self, min_sim=0.5, k_nbr=128, is_main_process=False):
        ptr = int(self.queue_ptr)
        if ptr == 0:
            if is_main_process:
                print("****************************************************")
                print(f"[{dist.get_rank()}] Starting DDP-based parallel clustering...")
            
            with Timer(f"DDP clustering on rank {dist.get_rank()}", verbose=False):
                info_features_gpu = nn.functional.normalize(self.queue_emb_copy, dim=1)
                
                unique_coarse_labs = torch.unique(self.queue_coarse_lab_copy)
                world_size = dist.get_world_size()
                rank = dist.get_rank()

                local_info_label = torch.zeros_like(self.info_label)
                base = 0 

                my_coarse_labs = [c_lab for i, c_lab in enumerate(unique_coarse_labs) if i % world_size == rank]

                for c_lab_tensor in my_coarse_labs:
                    c_lab = c_lab_tensor.item()
                    
                    mask = (self.queue_coarse_lab_copy == c_lab)
                    
                    info_feature_subset = info_features_gpu[mask].cpu().numpy()
                    
                    if info_feature_subset.shape[0] <= k_nbr:
                        pred_labels = np.zeros(info_feature_subset.shape[0], dtype=np.int32)
                    else:
                        
                        dists, nbrs = get_dist_nbr(feature=info_feature_subset, 
                                                k_nbr=k_nbr, 
                                                knn_method="faiss-gpu", 
                                                verbose=False)

                       
                        pred_labels, _, _, _ = cluster_by_infomap(nbrs, dists, min_sim=min_sim, verbose=False)

                    local_info_label[mask] = torch.from_numpy(pred_labels).int().cuda()

            dist.all_reduce(local_info_label, op=dist.ReduceOp.SUM)

            if is_main_process:
                final_info_label = torch.zeros_like(local_info_label)
                base = 0
                for c_lab_tensor in unique_coarse_labs:
                    mask = (self.queue_coarse_lab_copy == c_lab_tensor.item())
                    
                    class_labels = local_info_label[mask]
                    unique_class_labels = torch.unique(class_labels)
                    num_classes_labels = len(unique_class_labels)
                    if num_classes_labels > 0:
                        mapping = {old_label.item(): new_label for new_label, old_label in enumerate(unique_class_labels)}
                        remapped_labels = torch.tensor([mapping[l.item()] for l in class_labels], dtype=torch.int).cuda()
                        
                        final_info_label[mask] = base + remapped_labels
                        base += len(unique_class_labels)
                    
                self.info_label.data.copy_(final_info_label)
                print(f"Total {base} info fine classes found across all GPUs.")
                print("****************************************************")

            dist.barrier()
            
            dist.broadcast(self.info_label, src=0, async_op=False)


    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        batch_size_this = x.shape[0] 
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0] 
        num_gpus = batch_size_all // batch_size_this  

        idx_shuffle = torch.randperm(batch_size_all).cuda() 

        dist.broadcast(idx_shuffle, src=0)

        idx_unshuffle = torch.argsort(idx_shuffle)

    
        gpu_idx = dist.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        gpu_idx = dist.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, coarse_labs, args=None, is_train=True):
        coarse_cls_logit, q = self.encoder_q(im_q) 
        q = nn.functional.normalize(q, dim=1) 
        with torch.no_grad(): 
            self._momentum_update_key_encoder() 
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            _, k = self.encoder_k(im_k)  
            k = nn.functional.normalize(k, dim=1)
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1) 
        l_neg = torch.einsum("nc,kc->nk", [q, self.queue_emb.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1) 
        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        if is_train:
            with torch.no_grad():
                self._dequeue_and_enqueue(keys_emb=k, keys_coarse_lab=coarse_labs)
                self._info_cluster(min_sim=args.min_sim, k_nbr=args.k_nbr, is_main_process=misc.is_main_process())

        dino_loss = self.dino_loss_criterion(q, self.queue_emb_copy, self.info_label)
        cls_loss = self.cls_loss_criterion(coarse_cls_logit, coarse_labs)

        return logits, labels, dino_loss, cls_loss


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

class DINOLoss(nn.Module):
    def __init__(self, dino_dim=128):
        super().__init__()
        self.dino_dim = dino_dim

    def forward(self, batch_feature, queue_emb_copy, info_label):
        num_infocalss = np.max(info_label.cpu().numpy()) + 1
        centroids = torch.cat([torch.mean(queue_emb_copy[torch.where(info_label == idx)], dim=0, keepdim=True)
                               for idx in range(num_infocalss)], dim=0)
        feature_norm = batch_feature  
        centroids_norm = nn.functional.normalize(centroids, dim=1)
        similarity_matrix_cls = torch.mm(feature_norm, centroids_norm.T)
        pseudo_label = similarity_matrix_cls.max(dim=1)[1]
        mask = (pseudo_label.unsqueeze(-1) == info_label.unsqueeze(0)).detach_()

        sim_matrix = torch.einsum("bc,kc->bk", [batch_feature, queue_emb_copy.detach()])
        MSE_matrix = 2 - 2.0 * sim_matrix
   
        MAE_matrix = (MSE_matrix + 1e-6).sqrt()   

        min_entropy_loss = ((MAE_matrix * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)).mean()

        mask_converse = mask.logical_not()
      
        avg_inter_cluster_dist = ((MAE_matrix * mask_converse).sum(dim=1) / (mask_converse.sum(dim=1) + 1e-6)).mean()
    
        max_entropy_loss = 2.0 - avg_inter_cluster_dist

        loss = min_entropy_loss + max_entropy_loss

        return loss

@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ] 

    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
