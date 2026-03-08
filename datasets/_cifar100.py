from typing import Tuple, NoReturn, Optional
import os
import sys
import pickle

from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

class cifar100(Dataset):

    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',   
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

    def __init__(self, root: str = './data/cifar-100',
                 train: bool = True,
                 transform = None,
                 download: bool = False):
        super(cifar100, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train  

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []          
        self.targets = []      

        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    
                    entry = pickle.load(f, encoding='latin1')
                
                self.data.append(entry['data'])
                self.targets.extend(entry['fine_labels'])  


        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  

        self._load_meta()

        self.cls_num = 100

    def _load_meta(self) -> NoReturn:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple:
        """
        Args:
            index (int): Index
        Returns:
            Tuple: image, class
        """
        img, target = Image.fromarray(self.data[index]), self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, index, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> NoReturn:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

