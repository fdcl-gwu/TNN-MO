import os
import cv2
import numpy as np
from scipy.linalg import logm, expm

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

# def project3Dto2D(Pw,Hc,K):
#     M = K @ Hc[0:3,0:4]
#     #perspective projection keypoints (image coordinates)
#     Pc = M @ Pw.T
#     pc = Pc[0:2,:]/Pc[2,:]
#     pc[0,:] = 640-pc[0:1]
#     return pc

def Tau2HwHc(Dir):
    Tau = np.loadtxt(Dir + 'Tau'+'.txt')
    for i, tau in enumerate(Tau):
        Hw = twist2Hom(tau)
        hw_save = np.reshape(Hw[0:3,0:4], 12)
        hw_save = np.expand_dims( hw_save, axis=0)

        Hc = Hw2Hc(Hw)
        hc_save = np.reshape(Hc[0:3,0:4], 12)
        hc_save = np.expand_dims( hc_save, axis=0)

        if i == 0:
            Hw_save = hw_save
            Hc_save = hc_save
        else:
            Hw_save = np.concatenate((Hw_save,hw_save),axis = 0)
            Hc_save = np.concatenate((Hc_save,hc_save),axis = 0)
    np.savetxt(Dir + 'Hw_save.txt',Hw_save)
    np.savetxt(Dir + 'Hc_save.txt',Hc_save)

def Hom2twist(H):
    # HomogMatrix2twist Convert 4x4 homogeneous matrix to twist coordinates
    '''
    Input:  H(4,4): Euclidean transformation matrix (rigid body motion)
    Output:
        twist(6,1): twist coordinates. Stack linear and angular parts [v;w]
        Observe that the same H might be represented by different twist vectors
        Here, twist(4:6) is a rotation vector with norm in [0,pi]
    '''
    se_matrix = logm(H)

    # careful for rotations of pi; the top 3x3 submatrix of the returned
    # se_matrix by logm is not skew-symmetric (bad).

    v = se_matrix[0:3,3]
    w = Matrix2Cross(se_matrix[0:3,0:3])

    twist = np.array([v[0],v[1],v[2],w[0],w[1],w[2]])

    return twist

def twist2Hom(twist):
    #twist2HomogMatrix Convert twist coordinates to 4x4 homogeneous matrix
    '''
     Input:
        -twist(6,1): twist coordinates. Stack linear and angular parts [v;w]
     Output:
        -H(4,4): Euclidean transformation matrix (rigid body motion)
    '''
    v = twist[0:3] #linear part
    v = np.array([[v[0]],[v[1]],[v[2]]])
    w = twist[3:6] #angular part

    se_matrix = np.concatenate((np.concatenate((Cross2Matrix(w), v), axis=1),np.zeros((1, 4))),axis=0) # Lie algebra matrix

    H = expm(se_matrix)

    return H

def RT2hom(R,T):
    H = np.identity(4)
    H[0:3,0:3] = R

    H[0,3] = T[0]
    H[1,3] = T[1]
    H[2,3] = T[2]

    return H
    
def RT2homRow(R,T):
    H = np.identity(4)
    H[0:3,0:3] = R

    H[0,3] = T[0]
    H[1,3] = T[1]
    H[2,3] = T[2]

    return np.expand_dims( np.reshape(H[0:3,0:4], 12), axis=0)
    
def H_Row2H(H_row):
    H = np.eye(4)
    H[0:3,0:4] =np.reshape(H_row, (3,4))
    return H
    
def Hom2RT(H):
    R = H[0:3,0:3]
    T=np.array([0,0,0])
    T[0] = H[0,3]
    T[1] = H[1,3]
    T[2] = H[2,3]
    return R, T 
    
def Cross2Matrix(s):

    S = np.zeros((3,3))

    S[0,1]=-s[2]
    S[0,2]=s[1]
    S[1,0]=s[2]
    S[1,2]=-s[0]
    S[2,0]=-s[1]
    S[2,1]=s[0]

    return S
    
def Matrix2Cross(S):
    return [-S[1,2], S[0,2], -S[0,1]]

def H_AB_to_H_BA(H_AB):  
    # A->B  to B->A
    # relative to camera frame
    # Pc = Rw^TPw + (-Rw^T Tw)
    # Hc = [Rw^T | -Rw^T Tw] = [Rc | Tc]
    H_BA = np.zeros((4,4))
    H_BA[0:3,0:3] = H_AB[0:3,0:3].T
    H_BA[0:3,3] = -H_AB[0:3,0:3].T @ H_AB[0:3,3]
    H_BA[3,3] = 1
    return H_BA

def Hw2Hc(Hw):
    Hc = np.zeros((4,4))
    Hc[0:3,0:3] = Hw[0:3,0:3].T
    Hc[0:3,3] = -Hw[0:3,0:3].T @ Hw[0:3,3]
    Hc[3,3] = 1
    return Hc

def Hc2Hw(Hc):
    Hw = np.zeros((4,4))
    Hw[0:3,0:3] = Hc[0:3,0:3].T
    Hw[0:3,3] = -Hc[0:3,0:3].T @ Hc[0:3,3]
    Hw[3,3] = 1
    return Hw
