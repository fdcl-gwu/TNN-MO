"""
COCO dataset which returns image_id for evaluation.
https://github.com/pranoyr/End-to-End-Trainable-Multi-Instance-Pose-Estimation-with-Transformers/blob/main/datasets/coco.py
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
from PIL import Image
from numpy.core.defchararray import array
import torch
from pycocotools.coco import COCO
import numpy as np
import torch.utils.data
import cv2

from pycocotools import mask as coco_mask

import datasets.transforms as T



class CocoDetection(torch.utils.data.Dataset):
    def __init__(self, root_path, data_dir, subfolder, transforms,  return_masks):
        super(CocoDetection, self).__init__()
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

        self.img_folder = data_dir
        
        self.coco=COCO(root_path / 'COCO_Train' / "train.json")

        imgIds = sorted(self.coco.getImgIds())

        self.all_imgIds = []
        for image_id in imgIds:
            if self.coco.getAnnIds(imgIds=image_id) == []:
                continue
      
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            target = self.coco.loadAnns(ann_ids)
            num_keypoints = [obj["num_keypoints"] for obj in target]
            if sum(num_keypoints) == 0:
                continue

            self.all_imgIds.append(image_id)
            

    def __len__(self):
        return len(self.all_imgIds)

    def __getitem__(self, idx):
        image_id = self.all_imgIds[idx]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        target = self.coco.loadAnns(ann_ids)

        target = {'image_id': image_id, 'annotations': target}

        img = Image.open(self.img_folder +"/"+ self.coco.loadImgs(image_id)[0]['file_name'])
        img, target = self.prepare(img, target)


        if self._transforms is not None:
            img, target = self._transforms(img, target)
        target["labels"] = target["labels"] - 1

       
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size


        img_array = np.array(image)
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(img_array)

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno if obj["num_keypoints"]!=0]
        classes = torch.tensor(classes, dtype=torch.int64)
        
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno if obj["num_keypoints"]!=0]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 2)

        keypoints3dw = None
        if anno and "keypoints3dw" in anno[0]:
            keypoints3dw = [obj["keypoints3dw"] for obj in anno if obj["num_keypoints"]!=0]
            keypoints3dw = torch.as_tensor(keypoints3dw, dtype=torch.float32)

        rotation6d = None
        if anno and "rotation6d" in anno[0]:
            rotation6d = [obj["rotation6d"] for obj in anno if obj["num_keypoints"]!=0]
            rotation6d = torch.as_tensor(rotation6d, dtype=torch.float32)
            
        translation = None
        if anno and "translation" in anno[0]:
            translation = [obj["translation"] for obj in anno if obj["num_keypoints"]!=0]
            translation = torch.as_tensor(translation, dtype=torch.float32)

        rot6Dtrans = None
        if anno and "rot6Dtrans" in anno[0]:
            rot6Dtrans = [obj["rot6Dtrans"] for obj in anno if obj["num_keypoints"]!=0]
            rot6Dtrans = torch.as_tensor(rot6Dtrans, dtype=torch.float32)

        Area = None
        if anno and "area" in anno[0]:
            Area = [obj["area"] for obj in anno if obj["num_keypoints"]!=0]
            Area = torch.as_tensor(Area, dtype=torch.float32)

        Radius = None
        if anno and "area" in anno[0]:
            Radius = [obj["radius"] for obj in anno if obj["num_keypoints"]!=0]
            Radius = torch.as_tensor(Radius, dtype=torch.float32)
            # print("len ",translation.size())
            # num_translation = translation.shape[0]
            # if num_translation:
            #     translation = keypoints.view(num_translation, -1, 2)
            #     print("view",translation.size())

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]

        if self.return_masks:
            masks = masks[keep]

        target = {}
        target["boxes"] = boxes
        
        target["labels"] = classes #- 1
        
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id

        if keypoints is not None:
            target["keypoints"] = keypoints

        if keypoints3dw is not None:
            target["keypoints3dw"] = keypoints3dw

        if rotation6d is not None:
            target["rotation6d"] = rotation6d

        if translation is not None:
            target["translation"] = translation

        if rot6Dtrans is not None:
            target["rot6Dtrans"] = rot6Dtrans

        if Area is not None:
            target["area"] = Area
        
        if Radius is not None:
            target["radius"] = Radius

        # for conversion to coco api
        #area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        #target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        del target['boxes']
 
        return image, target


def make_coco_transforms(image_set, num_keys):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], num_keys),
        #AddGaussianNoise(0.1, 0.08),
        #T.AugMix(),
        #T.GaussianBlur()
    ])


    if image_set == 'train':
        return T.Compose([normalize,
        ])


    if image_set == 'val':
        return T.Compose([normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path) 
    data_dir = args.data_dir + "Train"
    coco    = Path('COCO_Train/')
    subfolder = Path(args.subfolder)
    
    #print("COCO type dataset path :", root)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'object_keypoints'
    PATHS = {
        "train": (data_dir, root / coco / f'train.json'),
        "val": (data_dir, root / coco / f'val.json'),
    }
    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(root, data_dir, subfolder, transforms=make_coco_transforms(image_set, args.num_keys), return_masks=args.masks)
    return dataset
