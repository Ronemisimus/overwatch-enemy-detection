from roboflow import Roboflow
from pycocotools.coco import COCO
from PIL import Image
import torchvision.transforms as tfms
import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
import numpy as np

convert = lambda x: [x[0],x[1],x[0]+x[2],x[1]+x[3]]

def collate_fn(batch):
    # Separate images and targets
    images, targets = list(zip(*batch))
    
    # Pad and stack targets into a batch tensor
    boxes = [torch.tensor([convert(ann['bbox']) for ann in target]) for target in targets]
    labels = [torch.tensor([ann['category_id'] for ann in target]) for target in targets]
    
    targets = [{'boxes': image_boxes, 'labels': image_labels} for image_boxes, image_labels in zip(boxes,labels)]

    return images, targets


dataset_root = "Proj-F-25"
train_root = dataset_root+"/train"
valid_root = dataset_root+"/valid"
test_root = dataset_root+"/test"
anot_file= "/_annotations.coco.json"

class CocoDataset(Dataset):
    def __init__(self, coco_annotation_file, image_folder, transform=None):
        self.coco = COCO(coco_annotation_file)
        self.image_folder = image_folder
        self.transform = transform
        self.ids = []
        for img_id in self.coco.imgs:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if ann_ids:
                self.ids.append(img_id)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):    
        img_id = self.ids[idx]
        img_path = os.path.join(self.image_folder, self.coco.imgs[img_id]['file_name'])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)
        return img, target


def download_dataset():
    if not os.path.isdir(dataset_root):
        rf = Roboflow(api_key="HGKjm1s8DncqgYUCOPeC")
        project = rf.workspace("untitled-project-obum0").project("proj-f")
        project.version(25).download("coco")

def build_dataset(small:bool):
    download_dataset()
    transform = tfms.ToTensor()
    train = CocoDataset(train_root+anot_file,train_root,transform=transform)
    valid = CocoDataset(valid_root+anot_file,valid_root,transform=transform)
    test = CocoDataset(test_root+anot_file,test_root,transform=transform)
    if small:
        train=Subset(train,np.random.randint(0,len(train),size=(100,)))
        valid=Subset(valid,np.random.randint(0,len(train),size=(25,)))
        test=Subset(test,np.random.randint(0,len(train),size=(25,)))
    return train,valid,test

def get_dataLoader(dataset,batch_size,shuffle):
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,collate_fn=collate_fn)
