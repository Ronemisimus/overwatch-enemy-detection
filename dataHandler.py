from roboflow import Roboflow
from pycocotools.coco import COCO
from PIL import Image
import torchvision.transforms as tfms
import SSDTransfoms as ssd_tfms
import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
import numpy as np

convert = lambda x: [x[0],x[1],x[0]+x[2],x[1]+x[3]]

def collate_fn(batch):
    # Separate images and targets
    images, targets = list(zip(*batch))
    


    return images, targets


dataset_root = "Goats-5"
train_root = dataset_root+"/train"
valid_root = dataset_root+"/valid"
test_root = dataset_root+"/test"
anot_file= "/_annotations.coco.json"

class DataAug(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.crop = tfms.RandomCrop((320,320))

    def forward(self, image,target):
        p = float(torch.rand((1,)))
        if p<1/3:
            return image,target
        elif p<2/3:
            C,H,W = image.shape
            y_range =np.arange(0,H,int(H/30))
            x_range = np.arange(0,W,int(W/30))

            x1,x2 = np.random.choice(x_range,(2,),replace=False)
            y1,y2 = np.random.choice(y_range,(2,),replace=False)
            x1, x2 = min(x1,x2), max(x1,x2)
            y1,y2 = min(y1,y2), max(y1,y2)
            image = image[:,y1:y2,x1:x2]
            convert = lambda box: torch.tensor([max(box[0]-x1,0),max(box[1]-y1,0),max(box[2]+min(box[0]-x1,0),0),max(box[3]+min(box[1]-y1,0),0)])
            for i, anot in enumerate(target):
                anot['bbox'] = convert(anot['bbox'])    

        else:
            jaccard_min = float(np.random.choice([0.1,0.3,0.5,0.7,0.9],(1,)))
            C,H,W = image.shape
            y_range =np.arange(0,H,int(H/30))
            x_range = np.arange(0,W,int(W/30))

            x1,x2 = np.random.choice(x_range,(2,),replace=False)
            y1,y2 = np.random.choice(y_range,(2,),replace=False)
            x1, x2 = min(x1,x2), max(x1,x2)
            y1,y2 = min(y1,y2), max(y1,y2)
            image = image[:,y1:y2,x1:x2]
            convert = lambda box: torch.tensor([max(box[0]-x1,0),max(box[1]-y1,0),max(box[2]+min(box[0]-x1,0),0),max(box[3]+min(box[1]-y1,0),0)])
            for i, anot in enumerate(target):
                anot['bbox'] = convert(anot['bbox']) 
            



class CocoDataset(Dataset):
    def __init__(self, coco_annotation_file, image_folder, transforms=None,scale_factor=1):
        self.coco = COCO(coco_annotation_file)
        self.image_folder = image_folder
        self.ids = []
        for img_id in self.coco.imgs:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if ann_ids:
                self.ids.append(img_id)
        self.transforms = transforms
        if transforms:
            self.transforms_choises = np.random.choice(range(len(transforms)),size=(len(self.ids)*scale_factor,),replace=True)
            self.img_choises = np.array(self.ids * scale_factor)
            np.random.shuffle(self.img_choises)
        else:
            self.transforms_choises = np.zeros((len(self.ids)*scale_factor,))
            self.img_choises = self.ids



    def __len__(self):
        return len(self.transforms_choises)

    def __getitem__(self, idx):
        tsfm_id = self.transforms_choises[idx]    
        img_id = self.img_choises[idx]
        img_path = os.path.join(self.image_folder, self.coco.imgs[img_id]['file_name'])
        img = Image.open(img_path).convert('RGB')
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        targets = self.coco.loadAnns(ann_ids)
        boxes = torch.tensor([convert(target['bbox']) for target in targets]).reshape(-1,4).to(torch.float)
        labels = torch.tensor([target['category_id'] for target in targets])
        target = {'boxes': boxes, 'labels': labels}
        if self.transforms:
            img, target = self.transforms[tsfm_id](img,target)
        return img, target

    def categories(self):
        return len(self.coco.cats)


def download_dataset():
    if not os.path.isdir(dataset_root):
        rf = Roboflow(api_key="rf_SS1TSoivnwRHQAMXcXt3ShNcc5x1")
        project = rf.workspace("justin-burger").project("goats-hqnax")
        dataset = project.version(5).download("coco")
        return dataset.name+"-"+dataset.version
        

def build_dataset(small:bool, scale):
    dataset_root = download_dataset()
    transforms = [
        ssd_tfms.Compose([
            ssd_tfms.PILToTensor(),
            ssd_tfms.ConvertImageDtype(torch.float),
            ssd_tfms.RandomIoUCrop(),
            ssd_tfms.ScaleToStandard()
        ]),
        ssd_tfms.Compose([
            ssd_tfms.PILToTensor(),
            ssd_tfms.ConvertImageDtype(torch.float),
            ssd_tfms.ScaleJitter((1920,1080)),
            ssd_tfms.ScaleToStandard()
        ]),
        ssd_tfms.Compose([
            ssd_tfms.PILToTensor(),
            ssd_tfms.ConvertImageDtype(torch.float),
            ssd_tfms.NoChange(),
            ssd_tfms.ScaleToStandard()
        ])
    ]
    test_transform = [
        ssd_tfms.Compose([
            ssd_tfms.PILToTensor(),
            ssd_tfms.ConvertImageDtype(torch.float),
            ssd_tfms.ScaleToStandard()
        ])
    ]
    train = CocoDataset(train_root+anot_file,train_root,transforms=transforms,scale_factor=scale)
    valid = CocoDataset(valid_root+anot_file,valid_root,transforms=test_transform)
    test = CocoDataset(test_root+anot_file,test_root,transforms=test_transform)
    categories = test.categories()
    if small:
        train=Subset(train,np.random.randint(0,len(train),size=(512,)))
        valid=Subset(valid,np.random.randint(0,len(valid),size=(256,)))
        test=Subset(test,np.random.randint(0,len(test),size=(256,)))
    return train,valid,test, categories

def get_dataLoader(dataset,batch_size,shuffle):
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,collate_fn=collate_fn)
