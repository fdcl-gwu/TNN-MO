from PIL import Image
import cv2
from cv2 import imshow
import numpy as np
import requests
import matplotlib.pyplot as plt
import os
# from plotlib.tools.projection import World3DInto2DImage, project3Dto2D
# from plotlib.tools.plot import plot_est_axis, plot_gt_axis, plot_points, plot_box
from tools import knownWorld3dBoxPoints, projectedKnown3dBoxPoints, pointsOnALine, cat_points
from tools.misc import *
Dir = os.getcwd()+"/"

def gen_pose_from_keypoints(Test_no=6,cat_id=1):
    cam_pop_folder = "CameraProp/Alvium_camera/"
    Pred_Keypoints = "Pred_Keypoints/Pc_Test_{}.txt".format(Test_no)
    Pcs = np.loadtxt(Dir + Pred_Keypoints)


    K = np.loadtxt(Dir + cam_pop_folder + "K.txt")
    distortion_coeffs = np.loadtxt(Dir + cam_pop_folder + "dist.txt")

    cat_id = 1
    P = cat_points(cat_id)

    for i in range(Pcs.shape[0]):
        Pw, Pw_C, Pw_I, Pw_axis, Pw_shifted_axis = knownWorld3dBoxPoints(P, points_only=0)
        p_C = Pcs[i,:].reshape(32,2)
        """Pose Estimation"""
        p_W = Pw.astype('double')
        distortion_coeffs = np.zeros((4,1))
        success, vector_rotation, vector_translation = cv2.solvePnP(p_W, p_C, K, distortion_coeffs, flags=0)
        T = vector_translation
        R, _ = cv2.Rodrigues(vector_rotation)

        """correction """
        tt = -180
        tt = tt * np.pi / 180
        R0 = np.array([[1,0,0],[0,np.cos(tt),-np.sin(tt)],[0,np.sin(tt),np.cos(tt)]]) #x axis
        #Hw = np.array([[np.cos(tt),0,np.sin(tt),0],[0,1,0,0],[-np.sin(tt),0,np.cos(tt),5],[0,0,0,1]]) #y axis
        #Hw = np.array([[np.cos(tt),-np.sin(tt),0,0],[np.sin(tt),np.cos(tt),0,0],[0,0,1,10],[0,0,0,1]])  #z axis
        H_pnp = np.concatenate((R,T),axis =1)
        Hc_est = R0@H_pnp

        # Hc_est = np.reshape(Hc_est, 12)
        # Hc_est = np.expand_dims(Hc_est, axis=0)
        H = np.eye(4)
        H[0:3,0:4] = Hc_est
        Hw_est = Hc2Hw(H)

        Hc_est = np.reshape(Hc_est, 12)
        Hc_est = np.expand_dims(Hc_est, axis=0)

        theta = 3.8*np.pi/180

        H = np.array([[np.cos(theta),-np.sin(theta),0,0],
                      [np.sin(theta),np.cos(theta),0,0],
                      [0,0,1,0],
                      [0,0,0,1]])

        Hw_est = H@Hw_est

        Hw_est = np.reshape(Hw_est[0:3,0:4], 12)

        Hw_est = removebias(Hw_est)

        Hw_est = np.expand_dims(Hw_est, axis=0)

        if i > 0:
            Hc = np.concatenate((Hc, Hc_est), axis = 0)
            Hw = np.concatenate((Hw, Hw_est), axis = 0)
        else:
            Hc = Hc_est
            Hw = Hw_est
            
        #np.savetxt(Main_Dir+ "pred/Hc_"+result+".txt",Hc)
        np.savetxt(Dir+ "Pred_Keypoints/Trajectory/Hw_Test_{}_gen.txt".format(Test_no),Hw)

def removebias(Hw_est):
    # m=0.22
    # n=0.5
    # d=2.0
    # bias_y = m*(np.tanh(n*np.abs(Hw_est[7])-d)+1)

    # Hw_est[7] = Hw_est[7]-bias_y
    
    m=0.47
    n=0.6
    d=2.5
    bias_z = m*(np.tanh(n*np.abs(Hw_est[7])-d)+1)
    Hw_est[11]=Hw_est[11]-bias_z

    return Hw_est
