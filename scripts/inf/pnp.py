import numpy as np
import cv2
import torch
from .convt import *

def simplePnP_Hc(Pw, Pc, K):
    distortion_coeffs = np.zeros((4,1))
    
    Pw = Pw.astype('double')
    Pc = Pc.astype('double')

    success, vector_rotation, vector_translation = cv2.solvePnP(Pw, Pc, K, distortion_coeffs, flags=0)
    T = vector_translation
    R, _ = cv2.Rodrigues(vector_rotation)

    # correction to blender camera axis
    R0 = np.array([[1.0, 0.0, 0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])

    tt = -180
    tt = tt * np.pi / 180
    R0 = np.array([[1,0,0],[0,np.cos(tt),-np.sin(tt)],[0,np.sin(tt),np.cos(tt)]])

    H_pnp = np.concatenate((R,T),axis =1)
    Hc_pnp = R0@H_pnp

    return Hc_pnp

def simplePnP_Hw(Pw, Pc, K):
    Hw_pnp = Hc2Hw3x4(simplePnP_Hc(Pw, Pc, K))
    return Hw_pnp
    

def batchPnP_Hc(Pws, Pcs, K):
    try:
        Pws = Pws.cpu().numpy()
        Pcs = Pcs.cpu().numpy()
    except:
        pass
    Hcs_pnp = np.array([simplePnP_Hc(Pws[i], Pcs[i], K) for i in range(Pcs.shape[0])])
    return Hcs_pnp

def batchPnP_Hw(Pws, Pcs, K):
    try:
        Pws = Pws.cpu().numpy()
        Pcs = Pcs.cpu().numpy()
    except:
        pass
    Hws_pnp = np.array([simplePnP_Hw(Pws[i], Pcs[i], K) for i in range(Pcs.shape[0])])
    return Hws_pnp

def batchPnP_PT(Pws, Pcs, K):
    try:
        Pws = Pws.cpu().numpy()
        Pcs = Pcs.cpu().numpy()
    except:
        pass
    Hcs_pnp = np.array([simplePnP_Hc(Pws[i], Pcs[i], K) for i in range(Pcs.shape[0])])
    Hws_pnp = np.array([Hc2Hw3x4(Hcs_pnp[i]) for i in range(Pcs.shape[0])])
    return torch.from_numpy(Hcs_pnp), torch.from_numpy(Hws_pnp)