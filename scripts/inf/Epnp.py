import torch
import torch.nn.functional as F
import numpy as np
import cv2
try:
    from pytorch3d.ops import efficient_pnp
    import pytorch3d.transforms as trfm
except:
    print("No module named 'pytorch3d'")

def Epnp(Pcs, Pws, K, dist = None, tensor=True, device= 'cpu', dtype=torch.float64):
    print("Pcs : ", Pcs.shape, " | Pws : ", Pws.shape, " | K : ", K.shape)
    """
    This function has Efficient PnP algorithm for Perspective-n-Points problem.
    It finds a camera position defined by rotation `R` and translation `T`, that
    minimizes re-projection error between the given 3D points `x` and
    the corresponding uncalibrated 2D points `y`.

    Opencv 'solvePnP' function is being used and opencv camera coordinate is used as dafault camera coordinate system
    """
    if torch.is_tensor(K):
        K = K.numpy()
    if len(Pcs.size()) == 3:
        H_pnp = []
        for i in range(Pcs.size()[0]):
            _, Rvec, Tvec = cv2.solvePnP(Pws[i].numpy(), Pcs[i].numpy(), K, distCoeffs=dist, flags=cv2.SOLVEPNP_EPNP)
            T = Tvec
            R, _ = cv2.Rodrigues(Rvec)
            h_pnp = np.concatenate((R,T),axis =1)
            h_pnp = np.expand_dims(h_pnp, axis = 0)
            if i == 0:
                H_pnp = h_pnp
            else:
                H_pnp = np.concatenate((H_pnp,h_pnp),axis =0)
        H_PnP = H_pnp

    if len(Pcs.size()) == 4:
        H_PnP = []
        for j in range(Pcs.size()[0]):
            H_pnp = []
            for i in range(Pcs.size()[1]):
                _, Rvec, Tvec = cv2.solvePnP(Pws[j][i].numpy(), Pcs[j][i].numpy(), K, distCoeffs=dist, flags=cv2.SOLVEPNP_EPNP)
                T = Tvec
                R, _ = cv2.Rodrigues(Rvec)
                h_pnp = np.concatenate((R,T),axis =1)
                h_pnp = np.expand_dims(h_pnp, axis = 0)
                if i == 0:
                    H_pnp = h_pnp
                else:
                    H_pnp = np.concatenate((H_pnp,h_pnp),axis =0)
            H_pnp = np.expand_dims(H_pnp, axis = 0)
            if j == 0:
                H_PnP = H_pnp
            else:
                H_PnP = np.concatenate((H_PnP,H_pnp),axis =0)
    if tensor:
        H_PnP = torch.from_numpy(H_PnP)
        H_PnP = H_PnP.to(device= device, dtype=dtype)

    return H_PnP

def EpnpPT(Pcs,Pws,k):
    """
    This function has Efficient PnP algorithm for Perspective-n-Points problem.
    It finds a camera position defined by rotation `R` and translation `T`, that
    minimizes re-projection error between the given 3D points `x` and
    the corresponding uncalibrated 2D points `y`.

    Pytorch3D 'efficient_pnp' function is being used and opencv camera coordinate system is used as camera coordinate system

    """

    if not torch.is_tensor(k):
        K = k.copy()
        K[0][0] = -k[0][0]
        K[1][1] = -k[1][1]
        K_tensor = torch.from_numpy(K).to(Pws.device, dtype=Pws.dtype)
    else:
        K = k.clone()
        K[0][0] = -k[0][0]
        K[1][1] = -k[1][1]
        K_tensor = K.to(dtype=Pws.dtype)
    K_inv = torch.inverse(K_tensor).T.unsqueeze(0)
    Pc3ds = torch.cat((Pcs, torch.ones(Pcs.size()[0], Pcs.size()[1], 1).to(Pcs.device)), dim=-1)
    Pc_2d_uncal = torch.matmul(Pc3ds, K_inv)
    transform = efficient_pnp(Pws, Pc_2d_uncal[...,:2])

    R = torch.mul(torch.transpose(transform.R,2,1), torch.tensor([[-1.0,-1.0,-1.0],[-1,-1,-1],[1,1,1]], dtype=Pws.dtype, device=Pcs.device))
    T = torch.mul(transform.T, torch.tensor([[-1.0,-1.0,1.0]], dtype=Pws.dtype, device=Pcs.device))
    '''
    **transform.x_cam**: Batch of transformed points `x` that is used to find
            the camera parameters, of shape `(minibatch, num_points, 3)`.
            In the general (noisy) case, they are not exactly equal to
            `x[i] R[i] + T[i]` but are some affine transform of `x[i]`s.
        **transform.R**: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        **transform.T**: Batch of translation vectors of shape `(minibatch, 3)`.
        **transform.err_2d**: Batch of mean 2D re-projection errors of shape
            `(minibatch,)`. Specifically, if `yhat` is the re-projection for
            the `i`-th batch element, it returns `sum_j norm(yhat_j - y_j)`
            where `j` iterates over points and `norm` denotes the L2 norm.
        **transform.err_3d**: Batch of mean algebraic errors of shape `(minibatch,)`.
            Specifically, those are squared distances between `x_world` and
            estimated points on the rays defined by `y`
    '''
    H_pnp =torch.cat((R,T.unsqueeze(2)), dim=-1)
   
    return H_pnp

def EpnptoHc(Pcs,Pws,k):
    '''change camera cordinate system'''
    return torch.matmul(torch.tensor([[1.0,0,0],[0,-1,0],[0,0,-1]], dtype=Pws.dtype, device=Pcs.device), EpnpPT(Pcs,Pws,k))

def HtoRT(H):
    '''
    input :  H is bsx3x4. H homogenious matrix which SE(3) matrix has the following form:  [ R ] [ T ]
    output : R is a bsx3x3 rotation matrix and output : T is a 3-D translation vector SE (3) matrices are commonly used to represent rigid motions or camera extrinsics
    '''
    R = H[:,:,:-1]
    T = H[:,:,3:4]
    return R, torch.transpose(T, 1,2).squeeze(1)

def Epnptojust6d(Pcs,Pws,k):
    '''original camera cordinate system'''
    R, T = HtoRT(EpnpPT(Pcs,Pws,k)) 
    return torch.cat((trfm.matrix_to_axis_angle(R), T), dim = -1)

def EpnptoHcpose6d(Pcs,Pws,k):
    R, T = HtoRT(EpnptoHc(Pcs,Pws,k))
    return torch.cat((trfm.matrix_to_axis_angle(R), T), dim = -1)

def project3Dto2D(Pws,Hcs,K):
    if not torch.is_tensor(K):
        K_tensor = torch.from_numpy(K).to(Pws.device) #.to(dtype=Hcs.dtype)
    else:
        
        K_tensor = K.to(dtype=Pws.dtype)
    Pws4d = torch.cat((Pws,torch.ones(Pws.size()[0],Pws.size()[1], 1).to(Pws.device)), dim=-1)
    M = torch.matmul(K_tensor,Hcs)
    Pws4d = torch.transpose(Pws4d, 2, 1)
    
    #perspective projection keypoints (image coordinates)
    Pcs3D = torch.matmul(M,Pws4d)
    Pcs2D = Pcs3D[...,:2,:]/Pcs3D[...,2,:].unsqueeze(1)
    Pcs2D[...,0,:]=K[0,2]*2-Pcs2D[...,0,:]
    Pcs2D_T = torch.transpose(Pcs2D, 2, 1)
    return Pcs2D_T
    
def Hc2Hw_tensor(Hc):
    Rc = Hc[...,0:3]
    RcT = torch.transpose(Rc, Hc.dim()-2, Hc.dim()-1)
    Pc = Hc[...,:,3].unsqueeze(Hc.dim()-1)
    Pw = torch.matmul(-RcT,Pc)
    Hw = torch.cat((RcT,Pw), dim =-1)
    return Hw

def reprojKeypoints(Pcs,Pws,k):
    '''
    1) Find pose from 2D Keypoints in the image
    2) Reprojected 3D Keypoints to the image using camera pose H
    '''
    return project3Dto2D(Pws, EpnptoHc(Pcs,Pws,k), k)