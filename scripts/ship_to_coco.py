import os
import shutil
import json
import numpy as np
#from glob import glob
#from pycocotools import mask as cocomask
from PIL import Image
from configparser import ConfigParser, ExtendedInterpolation
#from tools.mask2boxes import PILmask2bbox
#from tools.misc_fucntions import Tau2HwHc
from tools.rule import categorize_objects_2
from tools.plot import plot_cat_distribution, plot_cat_key_distribution
from tools import cat_names
import datetime
import argparse
from numpy.linalg import norm
from tqdm import tqdm 
import torch
import pytorch3d.transforms as tf

parser = argparse.ArgumentParser(description="Synthetic image generation")
parser.add_argument('--dataset_size', type=int, default=6000, help='Size of the dataset')
parser.add_argument('--subfolder', type=str, default="large", help='Subfolder')
parser.add_argument('--num_keys',default=32, type=int,help="Number of keypoints")
parser.add_argument('--points_only', default=0, type=int,help="Consider only points not a box")
parser.add_argument('--num_classes', default=1, type=int,help="Number of classes")
parser.add_argument('--main_dir', default=os.getcwd() + '/', type=str)
parser.add_argument('--data_dir', default=os.getcwd() + '/synthetic_dataset/offline_saved/', type=str)
parser.add_argument('--coco_path', default=os.getcwd() + '/synthetic_dataset/', type=str)
parser.add_argument('--configfile',default='config.ini', type=str)
args = parser.parse_args()

Total_Dataset = args.dataset_size  
subfolder = args.subfolder
num_keys = args.num_keys
Main_Dir = args.main_dir
data_dir = args.data_dir
coco_path = args.coco_path
num_clz = args.num_classes
configdir = args.configfile

config = ConfigParser(interpolation=ExtendedInterpolation())

file = args.configfile
config.read(file)
checkpoint_dir = config.get('Dir', 'CKPT')
ExpNo = config.get('Test', 'ExpNo')
RevNo = config.getint('Test', 'RevNo')
testname = config.get('Test', 'TESTNAME')

try:
    os.mkdir(checkpoint_dir+"/"+testname)
    print('Created Experiment E', ExpNo, ' folder ...')
    shutil.copy2(file, checkpoint_dir+"/"+testname+"/sconfig.ini")
except:
    print("path exist")

coco_path = checkpoint_dir+"/"+testname + "/"

config.read(checkpoint_dir+"/"+testname+"/sconfig.ini")


Total_Dataset = config.getint('Training', 'DATASET_SIZE')
subfolder = config.get('Dir', 'COCOSUBFOLDER')
num_keys = config.getint('DETR', 'NUM_KEYS')
Main_Dir = default=os.getcwd() + '/'
data_dir = Main_Dir + config.get('Dir', 'DATA_DIR')
# coco_path = config.get('Dir', 'COCO_PATH')
num_clz = config.getint('Test', 'NUM_CLASSES')

try:
    os.mkdir(coco_path)
except:
    pass

try:
    os.mkdir(coco_path+"COCO_Train/")
    os.mkdir(coco_path +"COCO_Train/dataset/")
except:
    pass


def seperateDataset(Total_Dataset, base_dir, data_dir):
    n_train = int(Total_Dataset*0.999)
    test_ratio = 0.008
    n_val = int((Total_Dataset-n_train)*(1-test_ratio))
    n_test = Total_Dataset-n_train-n_val

    #Tau = np.loadtxt(data_dir + 'Tau'+'.txt')
    Hc = np.loadtxt(data_dir + 'Hc'+'.txt')
    Hw = np.loadtxt(data_dir + 'Hw'+'.txt')
    #UV = np.loadtxt(data_dir + 'UV'+'.txt')
    try:
        file_train = open(base_dir + 'dataset/train.txt', 'x')
        file_val = open(base_dir + 'dataset/val.txt', 'x')
        file_test = open(base_dir + 'dataset/test.txt', 'x')
    except FileExistsError:
        file_train = open(base_dir + 'dataset/train.txt', 'w')
        file_val = open(base_dir +'dataset/val.txt', 'w')
        file_test = open(base_dir +'dataset/test.txt', 'w')

    for i_train in range(1,n_train+1): 
        file_train.write('{:06d}\n'.format(i_train))
    #np.savetxt(base_dir + 'dataset/'+'train'+'_Tau.txt',Tau[0:n_train,:])
    np.savetxt(base_dir + 'dataset/'+'train'+'_Hc.txt',Hc[0:n_train,:])
    np.savetxt(base_dir + 'dataset/'+'train'+'_Hw.txt',Hw[0:n_train,:])
    #np.savetxt(base_dir + 'dataset/'+'train'+'_UV.txt',UV[0:n_train,:])

    for i_val in range(1,n_val+1): 
        file_val.write('{:06d}\n'.format(i_val+n_train))
    #np.savetxt(base_dir + 'dataset/'+'val'+'_Tau.txt',Tau[n_train:n_train+n_val,:])
    np.savetxt(base_dir + 'dataset/'+'val'+'_Hc.txt',Hc[n_train:n_train+n_val,:])
    np.savetxt(base_dir + 'dataset/'+'val'+'_Hw.txt',Hw[n_train:n_train+n_val,:])
    #np.savetxt(base_dir + 'dataset/'+'val'+'_UV.txt',UV[n_train:n_train+n_val,:])
    for i_test in range(1,n_test+1): 
        file_test.write('{:06d}\n'.format(i_test+n_val+n_train))
    #np.savetxt(base_dir + 'dataset/'+'test'+'_Tau.txt',Tau[n_train+n_val:,:])
    np.savetxt(base_dir + 'dataset/'+'test'+'_Hc.txt',Hc[n_train+n_val:,:])
    np.savetxt(base_dir + 'dataset/'+'test'+'_Hw.txt',Hw[n_train+n_val:,:])
    #np.savetxt(base_dir + 'dataset/'+'test'+'_UV.txt',UV[n_train+n_val:,:])


Dir = coco_path + "COCO_Train/"

# Tau2HwHc(data_dir) #Hc_save
seperateDataset(Total_Dataset,Dir,data_dir)

MAX_N = 1

categories = [
    {
        "supercategory": "none",
        "name": "ship",
        "id": 0
    }
]

info = [{
        "description": "GWU FDCL YP boat synthetic Dataset",
        "url": "https://github.com/ManeeshW",
        "version": "0.2.0",
        "year": 2023,
        "contributor": "Maneesh",
        "dataset_size": Total_Dataset,
        "date_created": datetime.datetime.utcnow().isoformat(" "),
    }]
licenses = [
        {
            "id": 1,
            "name": "N/A",
            "url": "https://fdcl-gwu.github.io/website/",
        }
    ]

phases = ["train", "val"]
log_on = True
img_cat_bools = []
img_cat_logs = []
i = 0
for phase in phases:
    json_file = Dir + "{}.json".format(phase)
    with open(os.path.join(os.path.join(Dir + 'dataset/', phase + '.txt')), "r") as f:
        event_list = f.read().splitlines()

    res_file = {
        "info": info,
        "licenses": licenses,
        "categories": categories,
        "images": [],
        "annotations": [],
        "segmentation" : []
    }
    annot_count = 0
    image_id = 0
    processed = 0

    K = np.loadtxt(data_dir + "K.txt")
    Hc_all = np.loadtxt(data_dir + "Hc.txt")
    Hw_all = np.loadtxt(data_dir + "Hw.txt")
    
    #Hc_all = Hc_all.astype(int)

    for idx, name in enumerate(tqdm(event_list, desc ="Processing {} images...".format(phase))):
        i+=1
        idx = i
        ship_bbox = np.zeros(4) #PILmask2bbox(idx+1, Dir + data)

        img_path = os.path.join(data_dir +"Train/"+ "{:06d}.jpg".format(i))
        filename = os.path.join("{:06d}.jpg".format(i))
        #print(i, idx, name, filename, img_path)
        img = Image.open(img_path)
        img_w, img_h = img.size
        img_elem = {"file_name": filename,
                        "height": img_h,
                        "width": img_w,
                        "id": image_id}

        res_file["images"].append(img_elem)

        Hc = Hc_all[i-1,:].reshape(3,4)
        Tc = list(Hc[:,3])
        
        Hw = Hw_all[i-1,:].reshape(3,4)
        Tw = list(Hw[:,3])
        Rw = Hw[:,:3]
        Rw_tensor = torch.from_numpy(Rw)
        Rw_6Drep_tensor = tf.matrix_to_rotation_6d(Rw_tensor)
        Rw_6Drep = list(Rw_6Drep_tensor.numpy())
        
        Rw6D_Tw = list(np.concatenate((Rw_6Drep_tensor.numpy(),Tw),axis=0))

        # R = norm(Hc[:,3],2) #L2 norm


        
        Min_Thr_x = 20
        Min_Thr_y = 20
        
        cat_ids, Pcs, Pws, Areas, Radius, img_cat_bool_log, img_cat_key_log = categorize_objects_2(num_clz, Hc, K, Min_Thr_x, Min_Thr_y,points_only=0)

        if log_on:
            img_cat_bools_log = img_cat_bool_log
            img_cat_keys_log = img_cat_key_log
            log_on = False
        else:
            img_cat_bools_log = np.append(img_cat_bools_log, img_cat_bool_log, axis=0)
            img_cat_keys_log = np.append(img_cat_keys_log, img_cat_key_log, axis=0)

        for cat_id, Area, radius, Pc, Pw in zip(cat_ids, Areas, Radius, Pcs, Pws):
            ones = np.ones((3, 1)).reshape(-1) #class
            # Pc = np.concatenate((Pc, ones),axis = 1)
            Pc = Pc.astype(int)
            Pc = Pc.reshape(-1)
            Pc = Pc.tolist()

            Pw = Pw.reshape(-1)
            Pw = Pw.tolist()

            xmin = int(ship_bbox[0])
            ymin = int(ship_bbox[1])
            xmax = int(ship_bbox[2])
            ymax = int(ship_bbox[3])
            w = xmax - xmin
            h = xmax - ymin
            area = w * h
            poly = [[xmin, ymin],
                    [xmax, ymin],
                    [xmax, ymax],
                    [xmin, ymax]]

            annot_elem = {
                        "id": annot_count,
                        "bbox": [float(xmin),
                                float(ymin),
                                float(w),
                                float(h)],
                        "segmentation": list([poly]),
                        "num_keypoints" : num_keys,
                        "keypoints" : Pc,
                        "keypoints3dw" : Pw,
                        "len_tvec" : 3,
                        "rotation6d" : Rw_6Drep,
                        "translation" : Tw,
                        "rot6Dtrans" : Rw6D_Tw,
                        "image_id": image_id,
                        "ignore": 0,
                        "category_id": cat_id,
                        "iscrowd": 0,
                        "area": Area,
                        "radius": radius
                        }
            res_file["annotations"].append(annot_elem)
            annot_count += 1
        image_id += 1
        processed += 1
        
   
    with open(json_file, "w") as f:
        json_str = json.dumps(res_file)
        f.write(json_str)

    print('\033[96m'+"Processed {} images ".format(phase)+ '\033[0m')
print('\033[92m'+"<<< "+subfolder+" ship train/val data generation completed .. >>>"+'\033[0m')

np.savetxt(Dir + "img_cat_bools_log.txt",img_cat_bools_log.astype(int), fmt='%i')
np.savetxt(Dir + "img_cat_keys_log.txt",img_cat_keys_log.astype(int), fmt='%i')
plot_cat_distribution(cat_names, img_cat_bools_log, Dir)
plot_cat_key_distribution(cat_names, img_cat_keys_log, Dir)


