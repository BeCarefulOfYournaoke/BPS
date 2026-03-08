import os
from typing import Tuple,NoReturn
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive

class tinyimagenet(Dataset):
    url="http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename="tiny-imagenet-200.zip"
    base_folder="tiny-imagenet-200"

    def __init__(self,root: str = './data/tiny-imagenet',
                 train: bool = True,
                 transform=None,
                 download: bool = False):
        super(tinyimagenet,self).__init__()
        self.root=root
        self.transform=transform
        self.train=train

        if download:
            self.download()

        self.data=[]
        self.targets=[]

        if self.train:
            data_dir=os.path.join(self.root,self.base_folder,'train')
            classes=sorted(os.listdir(data_dir))
            self.classes=classes
            self.class_to_idx={cls_name:i for i,cls_name in enumerate(classes)}
            for cls_name in classes:
                img_dir=os.path.join(data_dir,cls_name,'images')
                for img_name in os.listdir(img_dir):
                    self.data.append(os.path.join(img_dir,img_name))
                    self.targets.append(self.class_to_idx[cls_name])
        else:
            val_dir=os.path.join(self.root,self.base_folder,'val')
            anno_file=os.path.join(val_dir,'val_annotations.txt')
            with open(anno_file,'r') as f:
                lines=f.readlines()

            img_to_cls={}
            classes=set()

            for line in lines:
                parts=line.strip().split('\t')
                img_name,cls_name=parts[0],parts[1]
                img_to_cls[img_name]=cls_name
                classes.add(cls_name)
            classes=sorted(list(classes))
            self.classes=classes
            self.class_to_idx={cls_name:i for i,cls_name in enumerate(classes)}

            img_dir=os.path.join(val_dir,'images')
            for img_name in os.listdir(img_dir):
                self.data.append(os.path.join(img_dir,img_name))
                self.targets.append(self.class_to_idx[img_to_cls[img_name]])

        self.cls_num=len(self.class_to_idx)

    def __getitem__(self, index:int) -> Tuple:
        img_path,target=self.data[index],self.targets[index]
        img=Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img=self.transform(img)

        return img,index,target

    def __len__(self) -> int:
        return len(self.data)

    def download(self) -> NoReturn:
        if os.path.exists(os.path.join(self.root,self.base_folder)):
            print("Tiny-ImageNet already exists.")
            return
        download_and_extract_archive(self.url,self.root,filename=self.filename)
