# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

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
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms as T


# class CocoDetection(torchvision.datasets.CocoDetection):
#     def __init__(self, img_folder, ann_file, transforms, return_masks):
#         super(CocoDetection, self).__init__(img_folder, ann_file)
#         self._transforms = transforms
#         self.prepare = ConvertCocoPolysToMask(return_masks)


#     def __getitem__(self, idx):
#         print(idx)
#         img, target = super(CocoDetection, self).__getitem__(idx)
#         image_id = self.ids[idx]
#         print(target)
#         print(image_id)
#         target = {'image_id': image_id, 'annotations': target}
#         img, target = self.prepare(img, target)
#         if self._transforms is not None:
#             img, target = self._transforms(img, target)
#         return img, target
# scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
# transforms_ = T.Compose([
#             T.RandomHorizontalFlip(),
#             T.RandomSelect(
#                 T.Compose([
#                 T.Rotate(0.5, [-25, 25]),
#                 T.RandomResize(scales, max_size=1333),
#                 ]),
#                 T.Compose([
#                     T.RandomResize([400, 500, 600]),
#                     T.RandomSizeCrop(384, 600),
#                     T.RandomResize(scales, max_size=1333),
#                 ])
#             ),
#            # normalize,
#         ])


class CocoDetection(torch.utils.data.Dataset):
    def __init__(self, root_path, transforms, return_masks):
        super(CocoDetection, self).__init__()
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

        self.img_folder = root_path / "offline_saved" /"Train"
        self.coco=COCO(root_path / "ship_coco" / "large" / "train.json")

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
        img = Image.open(self.img_folder / self.coco.loadImgs(image_id)[0]['file_name'])
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
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]

        if self.return_masks:
            masks = masks[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        del target['boxes']
 
        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
    sub = Path(args.subfolder)
    print(root)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'person_keypoints'
    PATHS = {
        "train": (root / "offline_saved/Train", root / "ship_coco" / sub/  f'train.json'),
        "val": (root / "offline_saved/Train", root / "ship_coco" / sub /f'val.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(root, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset