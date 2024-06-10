import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

def cameraLocFocus2Hom(cameraFocusPoint,cameraLocation):
    inv = 1
    OC = cameraLocation - cameraFocusPoint
    theta = rotAngleX(np.array([1,0,0]),np.array([OC[0],OC[1],0]),np.array([0,0,1]))
    # print(180*theta/np.pi)
    # print(np.cos(theta))
    Rz = np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    #print(Rz)
    x_hat = np.array([1,0,0]) @ Rz.T
    #print(x_hat)
    phi =  rotAngleZ(np.array([0,0,1]),OC,x_hat) 
    #print(180*phi/np.pi)
    Rx = np.array([[1,0,0],[0, np.cos(phi),-np.sin(phi)],[0,np.sin(phi),np.cos(phi)]])
    R = Rz @ Rx
    T = cameraLocation
    H = np.array([[R[0,0],R[0,1],R[0,2],T[0]],
              [R[1,0],R[1,1],R[1,2],T[1]],
              [R[2,0],R[2,1],R[2,2],T[2]],
              [0,0,0,1]])
    return H


def cal_pdf(ri,mu,sigma): #probability density function
    return (1/(sigma*np.sqrt(2*np.pi)))*np.exp((-1/2)*((ri-mu)/sigma)**2)

def cal_ni(ri,mu,sigma,Nc, Sum):
    return cal_pdf(ri,mu,sigma)*Nc/Sum

def randomCamLoc(ri,dr,mu,sigma,Nc,N,A,shift = [0,0,0]):
    theta_start = A[0]
    theta_end = A[1]
    theta = A[2]
    phi_start = A[3]
    phi_end = A[4]
    phi = A[5]
    Sum = 0
    for i in range(N):
        ri = i * dr
        Sum = cal_pdf(ri,mu,sigma) + Sum

    n = np.zeros(N)
    for i in range(N):
        ri = i * dr
        n[i] = int(cal_ni(ri,mu,sigma,Nc,Sum))

    ri = []
    thetai = []
    phii = []
    for i in range(N):
        ri = np.append(ri,(np.random.random(int(n[i]))*dr + i*dr)) 
        thetai = np.append(thetai,(np.random.random(int(n[i]))*theta + theta_start))
        phii = np.append(phii,(np.pi/2 - np.random.random(int(n[i]))*phi + phi_start))

    #Cartesian coordinates
    x = ri * np.sin(phii) * np.cos(thetai) + shift[0]
    y = ri * -np.sin(phii) * np.sin(thetai) + shift[1]
    z = ri * np.cos(phii) + shift[2]
    
    RCL = np.zeros((3,z.shape[0]))
    RCL[0,:] = x
    RCL[1,:] = y
    RCL[2,:] = z
    
    return RCL

def rotAngleX(a,b,n):
    sintheta = np.dot(n,np.cross(a, b))/(LA.norm(a)*LA.norm(b))
    costheta= np.dot(a,b)/(LA.norm(a)*LA.norm(b))
    if sintheta > 0 and costheta < 0:
        #print("Xone = ", np.arccos(costheta)*180/np.pi)
        return np.arccos(costheta) +np.pi/2
    elif sintheta < 0 and costheta < 0:
        #print("Xtwo = ", -np.arccos(costheta)*180/np.pi)
        return -np.arccos(costheta)+np.pi/2   
    elif sintheta < 0 and costheta >= 0:
        #print("Xthree = ", -np.arccos(costheta) *180/np.pi)
        return -np.arccos(costheta) + np.pi/2
    else:
        #print("Xfour = " , np.arccos(costheta)*180/np.pi)
        return np.arccos(costheta)+np.pi/2
    
def rotAngleZ(a,b,n):
    sintheta = np.dot(n,np.cross(a, b))/(LA.norm(a)*LA.norm(b))
    costheta= np.dot(a,b)/(LA.norm(a)*LA.norm(b))
    if sintheta > 0 and costheta < 0:
        #print("Zone = ", np.arccos(costheta)*180/np.pi)
        return np.arccos(costheta) +np.pi/2
    elif sintheta < 0 and costheta < 0:
        #print("Ztwo = ", -np.arccos(costheta)*180/np.pi)
        return -np.arccos(costheta)   
    elif sintheta < 0 and costheta >= 0:
        #print("Zthree = ", -np.arccos(costheta) *180/np.pi)
        return -np.arccos(costheta) + np.pi
    else:
        #print("Zfour = " , np.arccos(costheta)*180/np.pi)
        return np.arccos(costheta)