from PIL import Image
import cv2
from cv2 import imshow
import numpy as np
import requests
import matplotlib.pyplot as plt

import torch
from . import knownWorld3dBoxPoints, projectedKnown3dBoxPoints, pointsOnALine, cat_points, cat_name_ids
from .plot import *

def imgplot(image, Pc_preds, Hc_pred, K, cat_ids_good, colors):
	for i in range(Pc_preds.shape[0]):	
		"""Estimate pc """
		P_ = cat_points(cat_ids_good[i])
		Pc_est, _, _, Pc_est_axis, Pc_est_shifted_axis = projectedKnown3dBoxPoints(P_, Hc_pred[i],K, points_only=0)

		# draw 3D BBOX Lines
		plot_box(Pc_preds[i], one_clr = colors[cat_ids_good[i]-1])
		# draw 3D BBOX Area
		plotArea(Pc_preds[i], one_clr = colors[cat_ids_good[i]-1], alpha = 0.2) 
		# draw 3D BBOX corners
		plot_points(Pc_preds[i], clr = colors[cat_ids_good[i]-1])
		#plot_points(Pc_est, clr = "Darkred") #reproject
		plot_est_axis_oc(Pc_est_axis, colors[cat_ids_good[i]-1])

	plt.imshow(image)


def objplot(Pc_pred, color, size=30):
	plot_points(Pc_pred, clr = color, size=size)