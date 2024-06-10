import os
import cv2
import numpy as np
from scipy.linalg import logm, expm
import matplotlib.pyplot as plt

def World3DInto2DImage(Pw, Hc, K, Dir):
    Hc_all = Hc

    for i, hc in enumerate(Hc_all):
        Hc = np.reshape(hc, (3,4))
        #Perspective projection matrix M
        M = K @ Hc[0:3,0:4]
        #perspective projection keypoints (image coordinates)
        Pc = M @ Pw.T
        pc = Pc[0:2,:]/Pc[2,:]
        pc[0,:] = 640-pc[0:1]
        uv_save = np.reshape(pc, pc.shape[1]*2)
        uv_save = np.expand_dims( uv_save, axis=0)
        if i == 0:
            UV_save = uv_save
        else:
            UV_save = np.concatenate((UV_save,uv_save),axis = 0)
    np.savetxt(Dir + 'UV_save.txt',UV_save)

def project3Dto2D(Pw,Hc,K):
    ones = np.ones((Pw.shape[0],1))
    Pw = np.concatenate((Pw,ones), axis = 1)
    M = K @ Hc[0:3,0:4]
    #perspective projection keypoints (image coordinates)
    Pc = M @ Pw.T
    pc = Pc[0:2,:]/Pc[2,:]
    pc[0,:] = 640-pc[0:1]
    return pc.T

