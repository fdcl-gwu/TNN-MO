# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import cv2
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")
    
    if "keypoints" in target:
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        keypoints = target["keypoints"]
        cropped_keypoints = keypoints.view(-1, 3)[:,:2] - torch.as_tensor([j, i])
        cropped_keypoints = torch.min(cropped_keypoints, max_size)
        cropped_keypoints = cropped_keypoints.clamp(min=0)
        cropped_keypoints = torch.cat([cropped_keypoints, keypoints.view(-1, 3)[:,2].unsqueeze(1)], dim=1)
        target["keypoints"] = cropped_keypoints.view(target["keypoints"].shape[0], 17, 3)
        fields.append("keypoints")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes
    
    if "keypoints" in target:
        keypoints = target["keypoints"]
        keypoints[:,:,0] = w - keypoints[:,:, 0]
        target["keypoints"] = keypoints


    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target

# def aug_mix(image):
#     image = F.convert_image_dtype(image, dtype=torch.uint8)
#     augmenter = T.AugMix()
#     img = augmenter(image)
#     image = F.convert_image_dtype(img, dtype=torch.float32)
#     return image

# def Gauss_blur(image, kernel_size=(5, 9), sigma=(0.1, 5)):
#     image = F.convert_image_dtype(image, dtype=torch.uint8)
#     gauss = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
#     img = gauss(image)
#     image = F.convert_image_dtype(img, dtype=torch.float32)
#     return image

def rotate(image, target, angle):
    flipped_image = F.rotate(image, angle)

    w, h = image.size
    target = target.copy()
    
    if "keypoints" in target:
        keypoints = target["keypoints"][:,:,:2].view(-1,2) # num_keypoints, 2
        matrix = cv2.getRotationMatrix2D(((w - 1) * 0.5, (h - 1) * 0.5), angle, 1.0)

        keypoints = [torch.from_numpy(cv2.transform(np.array([[[keypoint[0], keypoint[1]]]]), matrix).squeeze()) for keypoint in keypoints]
        keypoints = torch.stack(keypoints, dim=0)
        v =  target["keypoints"].view(-1,3)[:,2].unsqueeze(1)
        keypoints = torch.cat([keypoints, v], dim=1).view(-1,17,3)
      
        target["keypoints"] = keypoints


    return flipped_image, target



def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes
    
    if "keypoints" in target:
        keypoints = target["keypoints"]
        scaled_keypoints = keypoints * torch.as_tensor([ratio_width, ratio_height, 1])
        target["keypoints"] = scaled_keypoints

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target

# class AugMix(object):
#     def __init__(self, p=0.5):
#         self.p = p

#     def __call__(self, img, target):
#         if random.random() < self.p:
#             return aug_mix(img), target
#         return img, target

# class GaussianBlur(object):
#     def __init__(self, p=0.5):
#         self.p = p

#     def __call__(self, img, target):
#         if random.random() < self.p:
#             return Gauss_blur(img, kernel_size=(5, 9), sigma=(0.1, 5)), target
#         return img, target

# class AddGaussianNoise(object):
#     def __init__(self, mean=0., std=1.):
#         self.std = std
#         self.mean = mean
        
#     def __call__(self, tensor):
#         return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
#     def __repr__(self):
#         return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std), target

class Rotate(object):
    def __init__(self, p=0.5, limit=[-25,25]):
        self.limit = limit
        self.p = p
    def __call__(self, img, target):
        if random.random() < self.p:
            angle = random.uniform(self.limit[0], self.limit[1])
            return rotate(img, target, angle)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std, num_keys):
        self.mean = mean
        self.std = std
        self.num_keys = num_keys
        

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        
        if "keypoints" in target:
            keypoints = target["keypoints"]  #  (2, 32, 3) (num_obj, num_keypoints, 3)
            Key = keypoints[:,:,:2]
            #print("key : ", Key.size())
            R = Key.view(-1, 2*self.num_keys)
            #print("R : ", R.size())
    
            #V = keypoints[:,:,2]        # visibility of the keypoints torch.Size([number of persons, 17])
            #V[V == 2] = 1
            #Z = keypoints.reshape(-1)
            # # get centers of keypoints
            # cxcy = (keypoints * V.unsqueeze(2)).sum(dim=1) / V.unsqueeze(2).repeat_interleave(3, dim=2).sum(dim=1)
            # cxcy = cxcy[:,:2] 
           
            # # get the distance between the center of the keypoints and the center of the image
            # cxcy_expand = cxcy.clone()
            # cxcy_expand = torch.repeat_interleave(cxcy_expand.unsqueeze(1) , 32, dim=1)
            # offsets = keypoints[:,:,:2] - cxcy_expand

            # C = cxcy                                # center of the keypoints  torch.Size([number of persons, 2])
            #Z = offsets.view(-1, 2*32)             # offsets of the keypoints torch.Size([number of persons, 32, 2]) --> n,34
            
            #C = C / torch.tensor([w, h], dtype=torch.float32)
    
            N = torch.tensor([w, h] * self.num_keys, dtype = torch.float32)
            # print("N : \n", N)
            
            Z = R / N 
            # print("Z : \n", Z)
            # print("ZN : \n", Z*N)
            #print(Z.size())
            #Z = torch.unsqueeze(Z,0)
            
            # torch.Size([1, 2])
            # torch.Size([1, 34])
            # torch.Size([1, 17])
            # torch.Size([1, 53])
            #all_keypoints = torch.cat([C, Z, V], dim=1)  # torch.Size([number of persons, 2+34+17]) # torch.Size([number of persons, 2+64+32])
            #all_keypoints = torch.cat([Z, V], dim=1)  # torch.Size([number of persons, 2+34+17]) # torch.Size([number of persons, 2+64+32])
            #print("Z : ", Z.size())
            
            target["keypoints"] = R / N 

        if "translation" in target:
            tvec =  target["translation"] #  (2, 32, 3) (obj_cat_id, num_keypoints, 3)
            #print("translation",translation.size())
            # tvec = translation[:,:,:2]
            #print("tvec",tvec.size())
            #cd = tvec.view(-1, 2*self.num_keys)
            #print("cd",cd.size())
            target["translation"] = tvec #.view(-1, 2*self.num_keys)

        if "keypoints3dw" in target:
            keypoints3dw =  target["keypoints3dw"] #  (2, 32, 3) (obj_cat_id, num_keypoints, 3)
            #print("translation",translation.size())
            # tvec = translation[:,:,:2]
            #print("tvec",tvec.size())
            #cd = tvec.view(-1, 2*self.num_keys)
            target["keypoints3dw"] = keypoints3dw

        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
