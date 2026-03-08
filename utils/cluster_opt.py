
import os
import numpy as np
import math
import faiss
import infomap
import time

class Timer():
    def __init__(self, name='task', verbose=True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose:
            print('[Time] {} consumes {:.4f} s'.format(
                self.name,
                time.time() - self.start))
        return exc_type is None


def intdict_2_ndarray(idx_2_label, default_val=-1):
    arr = np.zeros(len(idx_2_label)) + default_val
    for k, v in idx_2_label.items():
        arr[k] = v
    return arr

def knns_2_ordered_nbrs(knns, sort=True):
    if isinstance(knns, list):
        knns = np.array(knns)
    nbrs = knns[:, 0, :].astype(np.int32)
    dists = knns[:, 1, :]
    if sort:
        nb_idx = np.argsort(dists, axis=1)
        idxs = np.arange(nb_idx.shape[0]).reshape(-1, 1)
        dists = dists[idxs, nb_idx]
        nbrs = nbrs[idxs, nb_idx]
    return dists, nbrs


def get_links(single, links, nbrs, dists, min_sim):
    for i in range(nbrs.shape[0]):
        count = 0
        for j in range(0, len(nbrs[i])):
            if i == nbrs[i][j]:
                pass
            elif dists[i][j] <= 1 - min_sim:
                count += 1
                links[(i, nbrs[i][j])] = float(1 - dists[i][j])
            else:
                break
        if count == 0:
            single.append(i)
    return single, links


class Knn_faiss():
    def __init__(self, feature, k_nbr, index_path='', knn_method='faiss-gpu', verbose=True):
        self.verbose = verbose
        with Timer('[{}] build index with neighbour size-{}'.format(knn_method, k_nbr), verbose):
            if os.path.exists(index_path):
                print('[{}] read knns from {}'.format(knn_method, index_path))
                self.knns = np.load(index_path)['data']
            else:
                feature = feature.astype('float32')
                num, dim = feature.shape  

                if knn_method == 'faiss-gpu':
                    order = math.ceil(num/1e6)
                    if order > 1:
                        order = (order-1)*4
                    ngpus = faiss.get_num_gpus()
                    self.res = faiss.StandardGpuResources() 
                    self.res.setTempMemory(order * 1024 * 1024 * 1024)
                    index = faiss.GpuIndexFlatIP(self.res, dim)
                else:
                    index = faiss.IndexFlatIP(dim)
                index.add(feature)

        with Timer('[{}] query topk-{} instance'.format(knn_method, k_nbr), verbose):
            if os.path.exists(index_path):
                pass
            else:
                sims, nbrs = index.search(feature, k=k_nbr)
                self.knns_result = [(np.array(nbr, dtype=np.int32), 1 - np.array(sim, dtype=np.float32))
                             for nbr, sim in zip(nbrs, sims)]
    def freeGPU(self):
        self.res.noTempMemory()
        return

def read_meta(fn_meta, start_pos=0, verbose=True):
    label_2_idxs = {}
    idx_2_label = {}
    with open(fn_meta) as f:
        for idx, x in enumerate(f.readlines()[start_pos:]):
            label = int(x.strip())
            if label not in label_2_idxs:
                label_2_idxs[label] = []
            label_2_idxs[label].append(idx)
            idx_2_label[idx] = label
    inst_num = len(idx_2_label)
    cls_num = len(label_2_idxs)
    if verbose:
        print('read from label data with [{}] #cls: {}, #inst: {}'.format(fn_meta, cls_num, inst_num))
    return label_2_idxs, idx_2_label

def get_dist_nbr(feature, k_nbr=80, nproc=1, index_path='', knn_method='faiss-gpu', verbose=True):
    One_Knn = Knn_faiss(feature=feature, k_nbr=k_nbr, index_path=index_path, knn_method=knn_method, verbose=verbose)
    knns_result = One_Knn.knns_result
    dists, nbrs = knns_2_ordered_nbrs(knns_result)
    if knn_method=='faiss-gpu':
        One_Knn.freeGPU()
    return dists, nbrs 

def cluster_by_infomap(nbrs, dists, min_sim, label_path=None, metrics=[], pred_label_path=None, verbose=True):
    single = []
    links = {}

    with Timer('get links ', verbose=verbose):
        single, links = get_links(single=single, links=links, nbrs=nbrs, dists=dists, min_sim=min_sim)
    
    with Timer('add links to info ', verbose=verbose):
        infomapWrapper = infomap.Infomap("--two-level --directed")

        for (i, j), sim in links.items():
            _ = infomapWrapper.addLink(int(i), int(j), sim)

    infomapWrapper.run(silent=True, num_trials=3)

    label2idx = {}
    idx2label = {}

    for node in infomapWrapper.iterTree():
        if node.isLeaf():
            idx2label[node.physicalId] = node.moduleIndex()
            if node.moduleIndex() not in label2idx:
                label2idx[node.moduleIndex()] = []
            label2idx[node.moduleIndex()].append(node.physicalId)

    node_count = 0
    for k, v in label2idx.items():
        node_count += len(v)

    class_num = len(list(label2idx.keys()))

    for single_node in single:
        idx2label[single_node] = class_num
        label2idx[class_num] = [single_node]
        class_num += 1

    idx_len = len(list(idx2label.keys()))

    if pred_label_path is not None:
        with open(pred_label_path, 'w') as of:
            for idx in range(idx_len):
                of.write(str(idx2label[idx]) + '\n')
    
    pred_labels = intdict_2_ndarray(idx2label)
    
    return pred_labels, idx2label, class_num, idx_len

