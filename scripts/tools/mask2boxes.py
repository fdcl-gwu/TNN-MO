import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.ops import masks_to_boxes
from scipy.linalg import logm, expm

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# zero = torch.tensor([[]],dtype=torch.float32, device= "cuda:0")
def PILmask2bbox(name, Dir):
    img = Image.open(Dir+"TrainMask/{:06d}.png".format(name)).convert('RGB')
    img = np.asarray(img)
    img = predicted_colord_labels(img, labels = 'ship')
    convert_tensor = transforms.ToTensor()
    imgTensor = convert_tensor(img)
    bb = masks_to_boxes(imgTensor)
    return bb.numpy()[0]


def predicted_colord_labels(pred_mask, labels = 'ship'):
    if labels == 'ship':
        colors = [(128, 128, 192),]
    if labels == 'sea':
        colors = [(128, 0, 64),]
    if labels == 'sky':
        colors = [(128, 64, 0),]
    if labels == 'all':
        colors = [(128, 128, 192),(128, 0, 64),(128, 64, 0),]
  
    im = np.zeros(pred_mask.shape, np.uint8)
    img_color_labels = im.copy()

    for i,color in enumerate(colors):
        #color = (128, 128, 192)
        pred_label = np.where(pred_mask != color, img_color_labels, 255) #*255
        ret, pred_label= cv2.threshold(pred_label[:,:,1], 50, 255, cv2.THRESH_BINARY)
    return pred_label

def seg_best_output(prediction_probabilities): 
    #argmax along the channel dimension to get colord mask (0-ship,2-sea,3-sky)
    color_mask = torch.argmax(prediction_probabilities, dim=1).type(torch.float32)  
    #adding one
    color_mask = color_mask  + 1
    return color_mask.unsqueeze(1)

def masks_to_bboxes(color_mask, m = 'ship'):
    color_mask = color_mask.squeeze(1)
    #global device , zero
    zero = torch.zeros_like(color_mask, dtype=torch.float32, device="cuda:0" )
    t = torch.tensor(1,dtype=torch.float32, device="cuda:0") 
    if m == 'ship':
        mask = torch.tensor(1,dtype=torch.float32, device="cuda:0")
        mask = torch.where(color_mask != mask, zero, t)
        return masks_to_boxes(mask)
    if m == 'sea':
        mask = torch.tensor(2,dtype=torch.float32, device="cuda:0")
        mask = torch.where(color_mask != mask, zero, t)
        return masks_to_boxes(mask)
    if m == 'sky':
        mask = torch.tensor(3,dtype=torch.float32, device="cuda:0")
        mask = torch.where(color_mask != 3, zero, t)
        return masks_to_boxes(mask)

def clrmasks_to_mask(color_mask, m = 'ship'):
    color_mask = color_mask.squeeze(1)
    #global device , zero
    zero = torch.zeros_like(color_mask, dtype=torch.float32, device="cuda:0" )
    t = torch.tensor(1,dtype=torch.float32, device="cuda:0") 
    if m == 'ship':
        mask = torch.tensor(1,dtype=torch.float32, device="cuda:0")
        mask = torch.where(color_mask != mask, zero, t)
        return mask.unsqueeze(1)
    if m == 'sea':
        mask = torch.tensor(2,dtype=torch.float32, device="cuda:0")
        mask = torch.where(color_mask != mask, zero, t)
        return mask.unsqueeze(1)
    if m == 'sky':
        mask = torch.tensor(3,dtype=torch.float32, device="cuda:0")
        mask = torch.where(color_mask != 3, zero, t)
        return mask.unsqueeze(1)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # output = torch.tensor(()).to(device)
    # for i in range(a_org.size(0)):
    #     a = a_org[i,:,:]
    #     w_min = min(torch.where(a == 0)[1])
    #     h_min = min(torch.where(a == 0)[0])
    #     w_max = max(torch.where(a == 0)[1])
    #     h_max = max(torch.where(a == 0)[0])

    #     output = torch.cat((output,torch.Tensor([[w_min,h_min,w_max, h_max]]).to(device)), 0)  
    # return output

