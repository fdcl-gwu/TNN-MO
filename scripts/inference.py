from PIL import Image
import cv2
import json
from cv2 import imshow
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T

torch.set_grad_enabled(False);
import time

import sys
import os
sys.path.append(os.path.abspath(''))
#os.chdir("../")
Main_dir = os.getcwd()+"/"

from tools.misc_fucntions import *
from tools.projection import World3DInto2DImage, project3Dto2D
from tools.plot import *
#from tools import knownWorld3dBoxPoints, projectedKnown3dBoxPoints, pointsOnALine, cat_points, cat_name_ids
from tools.Key3d import keypoints3d, getPwsPT, getPwsPT_ordered
from tools.pnp import *
from tools.argsinf import args, colors, cat_names, results_dir, example_dir
from tools.implot import imgplot
from models import build_model
from inf.load import *

from configparser import ConfigParser, ExtendedInterpolation

def compute(N, Pws, colors, ext = ".jpg", zero_numbering = True, resize = False, num_keys = 32):
	if zero_numbering:
		print(example_dir +example+"/"+example+"_imgs/{:06d}".format(N)+ext)
		im = Image.open(example_dir +example+"/"+example+"_imgs/{:06d}".format(N)+ext)
		
	else:
		im = Image.open(example_dir +example+"/"+example+"_imgs/{:d}".format(N)+ext)

	if resize:
		newsize = (640, 480)
		im = im.resize(newsize)
	#start_t = time.perf_counter()

	outputs_new = model_pred(im, model, transform, num_keys, confidance = 0.9)

	Hc_all, Hw_all = batchPnP_PT(Pws, outputs_new["keypoints"], K)

	# t1 = time.perf_counter()

	
	image = np.array(im)
	cat_id_ordered = torch.tensor([i for i in range(1,outputs_new["keypoints"].size()[0]+1)])
	cat_ids_good = cat_id_ordered[outputs_new["rkeep"]]
	obj_confidance_all = outputs_new["confidance_values"]
	obj_confidance_inliers = obj_confidance_all[outputs_new["rkeep"]] #.cpu().numpy()
	Pc_preds = outputs_new["keypoints"][outputs_new["rkeep"]].cpu().numpy()
	Hc_inliers = Hc_all[outputs_new["rkeep"]].cpu().numpy()
	Hw_inliers = Hw_all[outputs_new["rkeep"]]
	
	#Hw_est_bayesian_fusion = bayesian_fusion(Hw_pred, confidance)
	
	
	softmax = nn.Softmax(dim=0)
	
	weights_inliers =  softmax(obj_confidance_inliers) #inliers weights 
	weights_all =  softmax(obj_confidance_all) #all weights 

	# with filtering objects by class confidance
	Hw_inliers_bf, Cov_P_inl, std_dev_P_inl, Cov_R_inl, std_dev_R_inl, U_inlier, D_inlier, V_inlier= bayesian_fusion(Hw_inliers, weights_inliers.cpu().numpy())
	Hw_inliers_BF= torch.from_numpy(np.expand_dims(Hw_inliers_bf, axis=0))
	print("--N-- > ",N , " img")
	print("Hw ([R|t] wrt ship) from Baysian Fusion : \n", Hw_inliers_BF)
	# (optional) without filtering objects
	# Hw_all_bf, Cov_P_all, std_dev_P_all, Cov_R_all, std_dev_R_all, U_all, D_all, V_all = bayesian_fusion(Hw_all, weights_all.cpu().numpy())
	# Hw_all_BF = torch.from_numpy(np.expand_dims(Hw_all_bf, axis=0))

	imgplot(image, Pc_preds, Hc_inliers, K, cat_ids_good, colors)
	#plt.savefig(results_dir + "results/"+result+"/pred_imgs/out_{:06d}.jpg".format(N))
	

	# Save the figure without white background and axis
	plt.savefig(results_dir + "results/"+result+"/pred_imgs/out_{:06d}.jpg".format(N), transparent=True, bbox_inches='tight', pad_inches=0)
	#plt.imsave(results_dir + "results/"+result+"/pred_imgs/out_{:06d}.jpg".format(N))

	Estimations = {"Inl_Weights_Hw" : {"weight_inliers" : weights_inliers, "Hw_inliers": Hw_inliers}, "Hw_inliers_bf": Hw_inliers_BF, 
				  "Cov_P_inliers": Cov_P_inl, "std_dev_P_inliers" : std_dev_P_inl, "Cov_R_inliers" : Cov_R_inl, "std_dev_R_inliers" : std_dev_R_inl,
				  "U_inliers": U_inlier, "D_inliers": D_inlier, "V_inliers": V_inlier,
		        #    "obj_confidance_all": obj_confidance_all, "Hw_all": Hw_all, "Hw_all_bf": Hw_all_BF, 
				#   "Cov_P_all": Cov_P_all, "std_dev_P_all" : std_dev_P_all, "Cov_R_all" : Cov_R_all, "std_dev_R_all" : std_dev_R_all,
				#   "U_all": U_all, "D_all": D_all, "V_all": V_all,
			       "Ids": cat_id_ordered, "Keep":  outputs_new["rkeep"]}

	return Estimations

HC= {}
HW ={}

def run(n, Pws, results_dir, result, colors, img_no = 1, ext = ".png",  gt= True, zero_numbering = True, resize = False, num_keys= 32):
	Estimations = []
	for j in range(img_no):
		fig, ax = plt.subplots(figsize=(640/80, 480/80), dpi=100) 
		fig.patch.set_visible(False)
		ax.axis('off')
		N = n+j
		try:
			Est= compute(N, Pws, colors, ext = ext, zero_numbering = zero_numbering, resize = resize, num_keys= num_keys)
			Estimations.append(Est)
		except:
			torch.save({"Estimations": Estimations},results_dir+ 'results/'+result+'/pts/Hw_'+result+'.pt')
			break

		plt.close(fig)

	torch.save({"Estimations": Estimations},results_dir+ 'results/'+result+'/pts/Hw_'+result+'.pt')


if __name__ == '__main__':
	config = ConfigParser(interpolation=ExtendedInterpolation())
	configdirlist = config.read('test_config.ini')
	examples_names = json.loads(config.get('Test', 'Test_samples'))

	for i, example in enumerate(examples_names):

		points_only = args.points_only
		num_keys = args.num_keys
		name = args.testname
		result = example+"_"+name
		ext = args.ext
		gt = args.gt
		zn = args.zero_numbering
		detr_version = args.detr_version
		nq = args.num_queries
		nc = args.num_classes

		Pws = getPwsPT_ordered(nq, nc, device='cuda:0', dtype=torch.float32)
		model, criterion, postprocessors = build_model(args)
		Dir = Main_dir + "examples/groundtruth/" + example + "/"

		print("Loading ... ", Main_dir + args.checkpoint_dir + args.testname + "/"+ args.snapshot)
		checkpoint = torch.load(Main_dir + args.checkpoint_dir + args.testname + "/"+ args.snapshot, map_location='cuda')
		model.load_state_dict(checkpoint["model"])

		# colors for visualization
		COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
				[0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

		"""
		DETR uses standard ImageNet normalization, and output boxes in relative image coordinates in $[x_{\text{center}}, y_{\text{center}}, w, h]$ format, 
		where $[x_{\text{center}}, y_{\text{center}}]$ is the predicted center of the bounding box, and $w, h$ its width and height. 
		Because the coordinates are relative to the image dimension and lies between $[0, 1]$, we convert predictions to absolute image coordinates and $[x_0, y_0, x_1, y_1]$ format for visualization purposes.
		"""

		# standard PyTorch mean-std input image normalization
		transform = T.Compose([
			T.ToTensor(),
			T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])

		#load [R|T] relative to camera frame (extrinsic parameters) 
		# load calibration matrix K (intrinsic)
		K = np.loadtxt(Main_dir + "synthetic_dataset/cam_prop/"+ "K_syn.txt")
		distortion_coeffs = np.loadtxt(Main_dir + "synthetic_dataset/cam_prop/" + "dist_syn.txt")
		gtDir = Main_dir + "examples/groundtruth/" + example + "/"

		try:
			Hw_all = np.loadtxt(gtDir + "Hw.txt")
			Hc_all = np.loadtxt(gtDir + "Hc.txt")
			P_cs = np.array(gtDir + "Hw.txt")
			Hc_all = np.loadtxt(gtDir + "Hc.txt")
		except:
			Hw_all = np.array([])
			Hc_all = np.array([])
			P_cs = np.array([])
			Hc_all = np.array([])

		#create result folder	
		try:
			os.mkdir(results_dir+ "results/")
		except:
			pass

		#create experiment folder	
		try:
			os.mkdir(results_dir+ "results/"+result)
			os.mkdir(results_dir+ "results/"+result+"/Pcs")
			os.mkdir(results_dir+ "results/"+result+"/pred_imgs/")
			os.mkdir(results_dir+ "results/"+result+"/pts")
			os.mkdir(results_dir+ "results/"+result+"/Hcs")
			os.mkdir(results_dir+ "results/"+result+"/Hws")
			os.mkdir(results_dir+ "results/"+result+"/trjectoryplots")
			
		except:
			pass

		N = 1
		run(N, Pws, results_dir, result, colors, img_no = 5000, ext = ext, gt= gt, zero_numbering = zn, resize = False, num_keys=num_keys)