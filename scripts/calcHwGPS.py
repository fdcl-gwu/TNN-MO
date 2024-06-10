import numpy as np
from numpy import linalg as L2
from tools.misc import *

def calHwfromGPS_cam(onr):
    for i in range(onr.rtk_X_camV2.shape[1]):
        if i == 0:
            T = onr.rtk_X_camV2[:,i]
            #ax.scatter(T[0],T[1],T[2], color="r", s=15)
            R_ROVER2XYZ = onr.Rot_ROVER_to_WorldXYZ2(onr.YPR_cam[:,i])
            R_CAMERA2XYZ = onr.Rot_ROVER_CAMERA_to_WorldXYZ(onr.YPR_cam[:,i])
            #RoverBox3D(ax, R = R_ROVER2XYZ, T=T, scale=2, ane1 = onr.ane1[:,i], ane2 = onr.ane2[:,i])
            Hw_Row = RT2homRow(R_CAMERA2XYZ,T)
            Hi = onr.Camera_Pose(onr.rtk_X_camV2[:,i], onr.Base_YPR_cam[:,i] , onr.YPR_cam[:,i]).reshape(1,12)
            #np.savetxt("/home/maneesh/Desktop/Syn_Ship_Data_1/Tracker/Hw.txt",Hw_Row)
        else:
            dist = L2.norm(T - onr.rtk_X_camV2[:,i])
            R_ROVER2XYZ = onr.Rot_ROVER_to_WorldXYZ2(onr.YPR_cam[:,i])
            R_CAMERA2XYZ = onr.Rot_ROVER_CAMERA_to_WorldXYZ(onr.YPR_cam[:,i])
            hw_Row = RT2homRow(R_CAMERA2XYZ,onr.rtk_X_camV2[:,i])
            hi = onr.Camera_Pose(onr.rtk_X_camV2[:,i], onr.Base_YPR_cam[:,i] , onr.YPR_cam[:,i]).reshape(1,12)
            Hw_Row = np.concatenate((Hw_Row,hw_Row),axis = 0)
            Hi = np.concatenate((Hi,hi), axis = 0)

    return Hw_Row, Hi

def calHwfromGPS(onr):
    for i in range(onr.rtk_X.shape[1]):
        if i == 0:
            T = onr.rtk_X[:,i]
            #ax.scatter(T[0],T[1],T[2], color="r", s=15)
            R_ROVER2XYZ = onr.Rot_ROVER_to_WorldXYZ2(onr.YPR[:,i])
            R_CAMERA2XYZ = onr.Rot_ROVER_CAMERA_to_WorldXYZ(onr.YPR[:,i])
            #RoverBox3D(ax, R = R_ROVER2XYZ, T=T, scale=2, ane1 = onr.ane1[:,i], ane2 = onr.ane2[:,i])
            Hw_Row = RT2homRow(R_CAMERA2XYZ,T)
            Hi = onr.Camera_Pose(onr.rtk_X[:,i], onr.Base_YPR[:,i] , onr.YPR[:,i]).reshape(1,12)
            #np.savetxt("/home/maneesh/Desktop/Syn_Ship_Data_1/Tracker/Hw.txt",Hw_Row)
        else:
            dist = L2.norm(T - onr.rtk_X[:,i])
            R_ROVER2XYZ = onr.Rot_ROVER_to_WorldXYZ2(onr.YPR[:,i])
            R_CAMERA2XYZ = onr.Rot_ROVER_CAMERA_to_WorldXYZ(onr.YPR[:,i])
            hw_Row = RT2homRow(R_CAMERA2XYZ,onr.rtk_X[:,i])
            hi = onr.Camera_Pose(onr.rtk_X[:,i], onr.Base_YPR[:,i] , onr.YPR[:,i]).reshape(1,12)
            Hw_Row = np.concatenate((Hw_Row,hw_Row),axis = 0)
            Hi = np.concatenate((Hi,hi), axis = 0)

    return Hw_Row, Hi


def calHwfromGPS_cam_orig(onr):
    for i in range(onr.rtk_X_cam.shape[1]):
        if i == 0:
            T = onr.rtk_X_cam[:,i]
            #ax.scatter(T[0],T[1],T[2], color="r", s=15)
            R_ROVER2XYZ = onr.Rot_ROVER_to_WorldXYZ2(onr.YPR_cam[:,i])
            R_CAMERA2XYZ = onr.Rot_ROVER_CAMERA_to_WorldXYZ(onr.YPR_cam[:,i])
            #RoverBox3D(ax, R = R_ROVER2XYZ, T=T, scale=2, ane1 = onr.ane1[:,i], ane2 = onr.ane2[:,i])
            Hw_Row = RT2homRow(R_CAMERA2XYZ,T)
            Hi = onr.Camera_Pose(onr.rtk_X_cam[:,i], onr.Base_YPR_cam[:,i] , onr.YPR_cam[:,i]).reshape(1,12)
            #np.savetxt("/home/maneesh/Desktop/Syn_Ship_Data_1/Tracker/Hw.txt",Hw_Row)
        else:
            dist = L2.norm(T - onr.rtk_X_cam[:,i])
            R_ROVER2XYZ = onr.Rot_ROVER_to_WorldXYZ2(onr.YPR_cam[:,i])
            R_CAMERA2XYZ = onr.Rot_ROVER_CAMERA_to_WorldXYZ(onr.YPR_cam[:,i])
            hw_Row = RT2homRow(R_CAMERA2XYZ,onr.rtk_X_cam[:,i])
            hi = onr.Camera_Pose(onr.rtk_X_cam[:,i], onr.Base_YPR_cam[:,i] , onr.YPR_cam[:,i]).reshape(1,12)
            Hw_Row = np.concatenate((Hw_Row,hw_Row),axis = 0)
            Hi = np.concatenate((Hi,hi), axis = 0)

    return Hw_Row, Hi