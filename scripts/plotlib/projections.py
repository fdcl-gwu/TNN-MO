import torch
import numpy as np


def project3Dto2D(Pws,Hcs,K):
    if not torch.is_tensor(K):
        K_tensor = torch.from_numpy(K).to(Pws.device, dtype=torch.float32)
    else:
        K_tensor = K
    Pws4d = torch.cat((Pws,torch.ones(Pws.size()[0],Pws.size()[1], 32, 1).to(Pws.device)), dim=-1)
    
    M = torch.matmul(K_tensor,Hcs)

    Pws4d = torch.transpose(Pws4d, 2, 3)
    
    #perspective projection keypoints (image coordinates)
    Pcs3D = torch.matmul(M,Pws4d)
    Pcs2D = Pcs3D[...,:2,:]/Pcs3D[...,2,:].unsqueeze(2)
    Pcs2D[...,0,:]=640-Pcs2D[...,0,:]
    Pcs2D_T = torch.transpose(Pcs2D, 2, 3)
    return Pcs2D_T
    
def Hc2Hw_tensor(Hc):
    Rc = Hc[...,0:3]
    RcT = torch.transpose(Rc, 2, 3)
    Pc = Hc[...,:,3].unsqueeze(3)
    Pw = torch.matmul(-RcT,Pc)
    Hw = torch.cat((RcT,Pw), dim =-1)
    return Hw
    
def Hc2Hw(Hc):
    Hw = np.zeros((3,3))
    Hw[0:3,0:3] = Hc[0:3,0:3].T
    Hw[0:3,3] = -Hc[0:3,0:3].T @ Hc[0:3,3]
    return Hw
