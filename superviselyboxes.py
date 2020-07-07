import os
import glob
import json
import torch
from torch.utils import data
from PIL import Image
from torchvision import transforms


class SuperviselyBoxes(data.Dataset):
    
    def __init__(self, root, dataset_name, transform=None):
        '''
            Root directory of downloaded supervisely datasets include:
            - dataset_name
            - - ann
            - - img
            - meta.json
            - obj_class_to_machine_color.json
        '''
        self.root = root
        self.name = dataset_name
        self.transform = transform
        self.filenames = []
        all_ann_files = glob.glob(os.path.join(root+dataset_name+'/ann/', '*.json'))
        # filter out images without annotations
        for json_path in all_ann_files:
            with open(json_path) as fs:
                json_suprv = json.load(fs)
                if len(json_suprv['objects']) > 0:
                    self.filenames.append(json_path)

        
    def __getitem__(self, index):
        ann_path = self.filenames[index]
        # loading image by changing the path:
        # before: /path/to/dataset/ann/example.png.json
        # after: /path/to/dataset/img/example.png
        img_path = ann_path[:-5].split('/')
        img_path[-2] = 'img'
        img_path = '/'.join(img_path)
        image = Image.open(img_path)
        image = transforms.ToTensor()(image)
        
        # loading supervisely file
        with open(ann_path) as fs:
            json_suprv = json.load(fs)

        # loading boxes
        rectangles = [obj for obj in json_suprv['objects'] if obj['geometryType'] == 'rectangle']
        rect_labels = {}
        i = 1
        for rect in rectangles:
            lab = rect['classTitle']
            if lab not in rect_labels.keys():
                rect_labels[lab] = i
                i = i+1

        # number of objects in the image
        num_objs = len(rectangles)

        # Bounding boxes for objects:
        # In supervisely format, rectangle in points exterior = [left, top],[right, bottom]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        for rect in rectangles:
            exterior = rect['points']['exterior']
            xmin = exterior[0][0]
            ymin = exterior[0][1]
            xmax = exterior[1][0]
            ymax = exterior[1][1]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(rect_labels[rect['classTitle']])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Tensorise img_id
        img_id = torch.tensor([index])
        
        # Areas
        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transform is not None:
            image = self.transform(image)

        return image, my_annotation
    
    
    def __len__(self):
        return len(self.filenames)
    
    