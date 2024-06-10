import torch
import numpy as np
from . import knownWorld3dBoxPoints, projectedKnown3dBoxPoints, cat_points

def genPw(p):
    Pw, _ , _ , _ , _ = knownWorld3dBoxPoints(p)
    return Pw

def genPc(p, Hc, K):
    Pc, _, _, _, _ = projectedKnown3dBoxPoints(p, Hc, K)
    return Pc

def getPws(cat_ids):
    Pws = np.array([genPw(cat_points(cat_ids[i])) for i in range(cat_ids.shape[0])])
    return Pws

def getPwsPT(cat_ids, NQ, NC, device='cuda:0', dtype=torch.float32):
    '''
        NQ = No of Queries
        NC = No of Classes
    '''
    
    print("NQ:", NQ)
    print("NC:", NC)
    print(cat_ids)
    print(np.array([i+1 if i < NC else 19 for i in range(NQ)]))
    Pws = np.array([genPw(cat_points(cat_ids[i])) if i < NC else genPw(cat_points(cat_ids[0])) for i in range(NQ)])
    PwsPT = torch.from_numpy(Pws).to(device=device, dtype=dtype)
    return PwsPT

def getPwsPT_ordered(NQ, NC, device='cuda:0', dtype=torch.float32):
    '''
        NQ = No of Queries
        NC = No of Classes
    '''
    print("cat id ", np.array([i+1 if i < NC else 0 for i in range(NQ)]))
    Pws = np.array([genPw(cat_points(i+1)) if i < NC else genPw(cat_points(0)) for i in range(NQ)])
    PwsPT = torch.from_numpy(Pws).to(device=device, dtype=dtype)
    return PwsPT

def keypoints3d(cat_ids, num_est_classes=11, num_keys=32, device='cuda:0', dtype=torch.float32):

    Kw3D = torch.empty((num_est_classes, num_keys, 3),device=device, dtype=dtype)
    for i in range(num_est_classes):
        if cat_ids[i]==0:
            cat_ids[i] = 1
        P = cat_points(cat_ids[i])

        Pw, Pw_C, Pw_I, Pw_axis, Pw_shifted_axis = knownWorld3dBoxPoints(P, points_only=0)
        Kw3D[i] = torch.from_numpy(Pw)
    return Kw3D
