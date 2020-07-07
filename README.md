# PyTorch dataset loader for Supervisely format
Code to create a PyTorch data loader for datasets in Supervisely format. To train Faster-RCNN as well as Keypoint-RCNN.

## Requirements
As of July 7th 2020 this code has been tested using:
| torch | torchvision | python |
|-------|-------------|--------|
| 1.4.0 | 0.5.0       | >=3.6  |

## Usage
<a href="https://supervise.ly">Supervisely</a> datasets are stored in the following folder structure:<br>

    /dataset_name/ann/img01.png.json
    /dataset_name/ann/img02.png.json
    /dataset_name/ann/...
    /dataset_name/img/img01.png
    /dataset_name/img/img02.png
    /dataset_name/img/...
    /dataset_name/meta.json
    /dataset_name/obj_class_to_machine_color.json

Keep this structure (only the two folders 'ann' and 'img' are needed).<br>

For bounding box detection with Faster-RCNN use:

```python
from superviselyboxes import SuperviselyBoxes
DATA_ROOT = '/path/to/data/'

# Initialize dataset
train_data = SuperviselyBoxes(root = DATA_ROOT, dataset_name = 'train', transform=None)

# Create data loader
train_data_loader = torch.utils.data.DataLoader(
    train_data, batch_size=4, shuffle=True, num_workers=4,
    collate_fn=collate_fn
)

# Fetch next batch
imgs, targets = next(iter(train_data_loader))
```

Note: collate_fn is taken from torch <a href="https://github.com/pytorch/vision/blob/master/references/detection/utils.py">vision/references/detection/utils.py</a><br>
Just use this code:

```python
def collate_fn(batch):
    return tuple(zip(*batch))
```

## Tutorial
In <a href="https://github.com/MarkusFox/supervisely-dataloader-fasterrcnn/blob/master/FasterRCNN-HowTo-Example.ipynb">FasterRCNN-HowTo-Example.ipynb</a> you find a full tutorial on how to use the dataloader to train a Faster-RCNN in PyTorch.<br>
It also includes code for visualization of the image and it's annotations.

## ToDos
+ Support loading mask annotations from polygon as well as bitmap geometryType
+ Making the loader work with arbitrary transforms
