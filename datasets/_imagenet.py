import os
import json
from PIL import Image
from torch.utils.data import Dataset

class imagenet(Dataset):

    legacy_img_file_dir = {
        "train": "ILSVRC2012_img_train",
        "val": "ILSVRC2012_img_val_",
    }
    standard_img_file_dir = {
        "train": "train",
        "val": "val",
    }
    
    all_object_name_file = "all_object_name.json"
    object_name2class_name_file = "object_name2class_name.json"


    def __init__(self, root, split="train", transform=None):
        assert split in ["train", "val"], 'arg "split" must in ["train", "val"]'
        self.root = root
        self.split = split
        self.transform = transform

        self.data = []
        self.targets = []

        self.img_file_dir = self._resolve_img_layout()

        self.all_object_name = self._get_all_object_name()
        self.classes = self.all_object_name
        self.all_class_name = self._get_all_class_name()
        self.object2idx, self.idx2object = self._get_object2idx_and_idx2object()
        self.class2idx, self.idx2class = self._get_class2idx_and_idx2class()

        self.cls_num = len(self.all_object_name)  

        self.samples = []
        self._get_samples()

    def _resolve_img_layout(self):
        legacy_train = os.path.join(self.root, self.legacy_img_file_dir["train"])
        standard_train = os.path.join(self.root, self.standard_img_file_dir["train"])

        if os.path.isdir(legacy_train):
            return dict(self.legacy_img_file_dir)
        if os.path.isdir(standard_train):
            return dict(self.standard_img_file_dir)

        raise FileNotFoundError(
            "ImageNet train directory not found. Expected one of: "
            f"'{self.legacy_img_file_dir['train']}' or '{self.standard_img_file_dir['train']}' "
            f"under root '{self.root}'."
        )

    def _get_all_object_name(self):
        metadata_file = os.path.join(self.root, imagenet.all_object_name_file)
        if os.path.isfile(metadata_file):
            with open(metadata_file, "r", encoding="utf-8") as f:
                all_object_name = json.load(f)
            if not isinstance(all_object_name, list):
                all_object_name = list(all_object_name)
            return all_object_name

        train_dir = os.path.join(self.root, self.img_file_dir["train"])
        object_names = sorted(
            [entry.name for entry in os.scandir(train_dir) if entry.is_dir()]
        )
        if not object_names:
            raise RuntimeError(f"No class folders found under '{train_dir}'.")
        return object_names

    def _get_all_class_name(self):
        metadata_file = os.path.join(self.root, imagenet.object_name2class_name_file)
        if not os.path.isfile(metadata_file):
            # Fallback: use synset folder name as class name.
            return list(self.all_object_name)

        with open(metadata_file, "r", encoding="utf-8") as f:
            object_name2class_name = json.load(f)
        if not isinstance(object_name2class_name, dict):
            object_name2class_name = dict(object_name2class_name)

        all_class_name = []
        for obj_nm in self.all_object_name:
            all_class_name.append(object_name2class_name.get(obj_nm, obj_nm))
        return all_class_name

    def _get_object2idx_and_idx2object(self):
        object2idx = {}
        idx2object = {}
        for i, obj_nm in enumerate(self.all_object_name):
            object2idx[obj_nm] = i
            idx2object[i] = obj_nm
        return object2idx, idx2object

    def _get_class2idx_and_idx2class(self):
        class2idx = {}
        idx2class = {}
        for i, cls_nm in enumerate(self.all_class_name):
            class2idx[cls_nm] = i
            idx2class[i] = cls_nm
        return class2idx, idx2class
    

    def _get_samples(self):
        img_file_dir = os.path.join(self.root, self.img_file_dir[self.split])
        if not os.path.isdir(img_file_dir):
            raise FileNotFoundError(
                f"ImageNet '{self.split}' directory not found at '{img_file_dir}'."
            )

        object_dirs = sorted([entry for entry in os.scandir(img_file_dir) if entry.is_dir()], key=lambda e: e.name)
        if not object_dirs:
            raise RuntimeError(
                f"Expected class-subfolder layout in '{img_file_dir}', but no subfolders were found. "
                "Please organize data as '<root>/<train|val>/<synset>/*.JPEG'."
            )

        for obj_nm in object_dirs:
            if obj_nm.name not in self.object2idx:
                continue

            obj_dir = os.path.join(img_file_dir, obj_nm.name)
            obj_idx = self.object2idx[obj_nm.name]

            img_files = sorted([entry for entry in os.scandir(obj_dir) if entry.is_file()], key=lambda e: e.name)
            for img_f in img_files:
                img_path = os.path.join(obj_dir, img_f.name)
                self.samples.append((img_path, obj_idx))
                self.data.append(img_path)
                self.targets.append(obj_idx)

        if not self.samples:
            raise RuntimeError(f"No images found under '{img_file_dir}'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, fine_lab = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, idx, fine_lab
