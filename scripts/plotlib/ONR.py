import numpy as np
from scipy.spatial.transform import Rotation as Rmat
import pandas as pd
import pymap3d as pm

class ONR:
    def __init__(self, df, base_yaw_init_correct= 0.0, cam_Incline_Angle = 1.4, T_BASO_ship = np.array([0.3, 2.92, 1.283]), CamPos = np.array([0.0, 0, 0.23]), yaw_init_corr = 0.0):
        # World XYZ frame {X- East, Y - North, Z - UP}
        # World NED frame {X- North, Y - East, Z - Down}
        # World NED with respective to World fixed XYZ frame = NED2XYZ
        self.R_NED2XYZ = np.array([[0.0, 1.0, 0.0],
                                   [1.0, 0.0, 0.0],
                                   [0.0, 0.0,-1.0]])
        # measured
        # T_BASO_NED = BaseAntena position to Ship origin with respective to NED frame 
        #self.T_BASO_ship = np.array([-1.82, 3.53, 2.39])
        self.T_BASO_ship = T_BASO_ship #np.array([0.3, 2.92, 1.283])#np.array([-1.82, 3.53, 2.39]) #Ship frame
        self.CamPos = CamPos #np.array([0.0, 0, 0.23]) #Ship frame

        #initialize
        self.base_yaw_init= 0
        self.yaw_init = 0
        self.base_yaw_init_corr= base_yaw_init_correct
        self.yaw_init_corr = yaw_init_corr
        self.CameraInclineAngle = cam_Incline_Angle

        
        self.t = self.get(df, 't') / 1000.0
        self.N = len(self.t)
        self.ypr = self.get_3x1(df, 'ypr') 
        self.a = self.get_3x1(df, 'a')
        self.a_mag_norm = self.cal_a_mag_norm()
        self.W = self.get_3x1(df, 'W')
        self.rtk_x = self.get_3x1(df, 'rtk_x') #NED frame
        
        self.rtk_v = self.get_3x1(df, 'rtk_v')
        self.llh = self.get_3x1(df, 'llh')
        self.gps_status = self.get(df, 'status')
        self.gps_num_sats = self.get(df, 'sats')
        self.ane1 = self.get_3x1(df, 'ane1')
        self.ane2 = self.get_3x1(df, 'ane2')

        self.base_ypr = self.get_3x1(df, 'base_ypr') 
        
        self.base_a = self.get_3x1(df, 'base_a')
        self.base_W = self.get_3x1(df, 'base_W')
        self.base_llh = self.get_3x1(df, 'base_llh')
        self.base_gps_status = self.get(df, 'base_status')
        self.base_gps_num_sats = self.get(df, 'base_sats')
        self.base_ane = self.get_3x1(df, 'base_ane')
        self.idx1 = self.get(df, 'idx1')
        self.idx2 = self.get(df, 'idx2')
        self.time = self.get(df,'t')

        self.InitialYawBase = self.base_yaw_from_llh(8) 
        self.InitialYawRover = self.Rover_yaw_from_RTK_v()
        self.Base_YPR = self.base_yaw_correction(self.base_ypr, self.InitialYawBase, self.base_yaw_init_corr)
        self.YPR = self.rover_yaw_correction_aligned(self.ypr) #self.base_yaw_correction(self.ypr, self.InitialYawRover)
        self.rtk_X = self.rtk_x2rtk_X() #Ship frame
        self.rtk_X_Y = self.rtk_x2rtk_X_Y() #Ship frame
        self.rtk_X_camV2 = self.get_nx3_correspond_imgs_V2(df, self.rtk_X)
        self.rtk_X_Y_camV2 = self.get_nx3_correspond_imgs_V2(df, self.rtk_X_Y)

        self.ypr_cam = self.get_nx3_correspond_imgs(df, 'ypr')
        self.YPR_cam = self.rover_yaw_correction_aligned(self.ypr_cam)
        self.a_cam = self.get_nx3_correspond_imgs(df, 'a')
        self.W_cam = self.get_nx3_correspond_imgs(df, 'W')
        self.llh_cam = self.get_nx3_correspond_imgs(df, 'llh')
        
        
        self.base_ypr_cam = self.get_nx3_correspond_imgs(df, 'base_ypr')
        self.Base_YPR_cam = self.base_yaw_correction(self.base_ypr_cam, self.InitialYawBase, self.base_yaw_init_corr)
        self.base_a_cam = self.get_nx3_correspond_imgs(df, 'base_a')
        self.base_W_cam = self.get_nx3_correspond_imgs(df, 'base_W')
        self.base_llh_cam = self.get_nx3_correspond_imgs(df, 'base_llh')

        self.rtk_x_cam, self.Time = self.get_nx3_correspond_imgs_with_time(df, 'rtk_x') #NED frame
        self.rtk_X_cam = self.rtk_x2rtk_X_cam() #Ship frame
        self.rtk_v_cam = self.get_nx3_correspond_imgs(df, 'rtk_v')

        self.maxPosVal = self.maxRTKvalues()
        # self.base_R = self.ypr_array_to_R_array(self.base_ypr)
        # self.R = self.ypr_array_to_R_array(self.ypr) 0, 2.915, 1.308

    def cal_a_mag_norm(self):
        a_magnitude = np.zeros(self.a.shape[1])
        for i in range(self.a.shape[1]):
            a_magnitude[i] = np.linalg.norm(self.a[:, i])

        return a_magnitude / np.max(a_magnitude)



    def set_base_antena_loc(self,base_loc=np.array([-0.7, 3.02, 1.233])):
        self.T_BASO_ship = base_loc
        self.rtk_X = self.rtk_x2rtk_X()
        self.rtk_X_cam = self.rtk_x2rtk_X_cam()
        self.rtk_X_Y = self.rtk_x2rtk_X_Y()  
        
    def correct_base_imu_yaw(self, yaw_corr):
        self.base_yaw_init_corr = yaw_corr
        
    def get_3x1(self, df, label):

        return np.vstack(
            (
                df['{}_0'.format(label)].values, 
                df['{}_1'.format(label)].values, 
                df['{}_2'.format(label)].values
            )
        )
    

    def get(self, df, label):
        try:
            return df[label].values
        except:
            return 0
    
    def get_nx3_correspond_imgs_with_time(self, df, label, cam_lable='idx2'):
        try:
            cam_idx = self.get(df, cam_lable)
            read_data = self.get_3x1(df, label)
            read_time = self.get(df,'t')
            data = np.array([[0,0,0]])
            time = np.array([[0]])
            for i in range(1,np.max(cam_idx)+1):
                try:
                    img_idx=np.where(cam_idx==i)[0][0]+1
                    data=np.expand_dims(read_data[:,img_idx], axis = 0)
                    time = np.array([[read_time[img_idx]]])
                except:
                    # print("cam index = ",i)
                    # print(data)
                    data=data
                    time=time

                if i == 1:
                    Data=data
                    Time=time
                else:
                    Data = np.concatenate((Data,data),axis=0)
                    Time = np.concatenate((Time,time),axis=0)

            return Data.T, Time
        except:
            return np.array([0,0,0]), np.array([0])
        
    def get_nx3_correspond_imgs(self, df, label, cam_lable='idx2'):
        try:
            cam_idx = self.get(df, cam_lable)
            read_data = self.get_3x1(df, label)
            data = np.array([[0,0,0]])
            for i in range(1,np.max(cam_idx)+1):
                try:
                    img_idx=np.where(cam_idx==i)[0][0]+1
                    data=np.expand_dims(read_data[:,img_idx], axis = 0)
                except:
                    # print("cam index = ",i)
                    # print(data)
                    data=data

                if i == 1:
                    Data=data
                else:
                    Data = np.concatenate((Data,data),axis=0)

            return Data.T
        except:
            return np.array([0,0,0])
        
    def get_nx3_correspond_imgs_V2(self, df, read_data3x1, cam_lable='idx2'):
        try:
            cam_idx = self.get(df, cam_lable)
            data = np.array([[0,0,0]])
            for i in range(1,np.max(cam_idx)+1):
                try:
                    img_idx=np.where(cam_idx==i)[0][0]+1
                    data=np.expand_dims(read_data3x1[:,img_idx], axis = 0)
                except:
                    # print("cam index = ",i)
                    # print(data)
                    data=data

                if i == 1:
                    Data=data
                else:
                    Data = np.concatenate((Data,data),axis=0)

            return Data.T
        except:
            return np.array([0,0,0])

    def Rot_POBI_to_WorldXYZ(self,base_ypr): 
        # Physical Orientation of Base IMU (POBI) measurements with repective to World fixed NED frame =  POBI2NED
        # Vector-nav IMU is used (Absolute mode NED (measure YPR angles deg respective to magnetic North pole))
        # R_POBI2NED = Rmat.from_euler('zyx', (-base_ypr[0],-base_ypr[1],base_ypr[2]), degrees=True).as_matrix() # YPR ++- / --+
        R_POBI2NED = self.ypr_to_R(base_ypr)

        # Physical Orientation of Base IMU (POBI) measurements with repective to World fixed XYZ frame =  POBI2XYZ
        R_POBI2XYZ = self.R_NED2XYZ @ R_POBI2NED

        return R_POBI2XYZ

    def Rot_SHIP_to_WorldXYZ(self,base_ypr):
        # Physical Orientation of Base IMU (POBI) with respective to Ship (Printed xyz-axes on IMU orientation respective to ship) = POBI2SHIP
        # {X_POBI-Bow, Y_POBI-Port Z_POBI-Up}
        R_POBI2SHIP = np.array([[0.0,-1.0,0.0],
                                [1.0, 0.0,0.0],
                                [0.0, 0.0,1.0]])
        
        R_POBI2XYZ = self.Rot_POBI_to_WorldXYZ(base_ypr)

        # Ship orientation with respective to world frame
        # Ship XYZ frame {X- Starboard, Y - Bow, Z - UP}
        R_SHIP2XYZ = R_POBI2XYZ @ R_POBI2SHIP.T

        return R_SHIP2XYZ
    
    def Rot_NED_to_SHIP(self,base_ypr):
        R_SHIP2XYZ = self.Rot_SHIP_to_WorldXYZ(base_ypr)

        R_NED2SHIP = R_SHIP2XYZ.T @ self.R_NED2XYZ 

        return R_NED2SHIP
    
    def Rot_POBI_to_WorldXYZ_yaw_only(self,base_ypr): 
        # Physical Orientation of Base IMU (POBI) measurements with repective to World fixed NED frame =  POBI2NED
        # Vector-nav IMU is used (Absolute mode NED (measure YPR angles deg respective to magnetic North pole))
        # R_POBI2NED = Rmat.from_euler('zyx', (-base_ypr[0],-base_ypr[1],base_ypr[2]), degrees=True).as_matrix() # YPR ++- / --+
        """
        base_ypr[1]=-1.5
        base_ypr[2]=-178.837997436523
        """
        R_POBI2NED = self.ypr_to_R(base_ypr)

        # Physical Orientation of Base IMU (POBI) measurements with repective to World fixed XYZ frame =  POBI2XYZ
        R_POBI2XYZ = self.R_NED2XYZ @ R_POBI2NED

        return R_POBI2XYZ
    
    def Rot_SHIP_to_WorldXYZ_Y(self,base_ypr):
        # Physical Orientation of Base IMU (POBI) with respective to Ship (Printed xyz-axes on IMU orientation respective to ship) = POBI2SHIP
        # {X_POBI-Bow, Y_POBI-Port Z_POBI-Up}
        R_POBI2SHIP = np.array([[0.0,-1.0,0.0],
                                [1.0, 0.0,0.0],
                                [0.0, 0.0,1.0]])
        
        R_POBI2XYZ = self.Rot_POBI_to_WorldXYZ_yaw_only(base_ypr)

        # Ship orientation with respective to world frame
        # Ship XYZ frame {X- Starboard, Y - Bow, Z - UP}
        R_SHIP2XYZ = R_POBI2XYZ @ R_POBI2SHIP.T

        return R_SHIP2XYZ
    
    def Rot_NED_to_SHIP_Y(self,base_ypr):
        R_SHIP2XYZ = self.Rot_SHIP_to_WorldXYZ_Y(base_ypr)

        R_NED2SHIP = R_SHIP2XYZ.T @ self.R_NED2XYZ 

        return R_NED2SHIP

    def Rot_PORI_to_WorldXYZ2(self, ypr):
        
        # Physical Orientation of Rover IMU (PORI) measurements with repective to World fixed NED frame =  PORI2NED
        # Vector-nav IMU is used (Absolute mode NED (measure YPR angles deg respective to magnetic North pole))
        # R_PORI2NED = Rmat.from_euler('zyx', (ypr[0],ypr[1],ypr[2]), degrees=True).as_matrix()
        R_PORI2NED = self.ypr_to_R(ypr) 

        # Physical Orientation of Rover IMU (PORI) measurements with repective to World fixed XYZ frame =  PORI2XYZ
        R_PORI2XYZ = self.R_NED2XYZ @ R_PORI2NED

        return R_PORI2XYZ

    def Rot_ROVER_to_WorldXYZ2(self, ypr):
        # Physical Orientation of Rover IMU(PORI) with respective to Ship (Printed xyz-axes on IMU orientation respective to ship) = PORI2SHIP
        # {X_PORI-Bow, Y_PORI-starboard, Z_PORI-Down}
        R_PORI2ROVER = np.array([[0.0, 1.0, 0.0],
                                 [1.0, 0.0, 0.0],
                                 [0.0, 0.0,-1.0]]) 
        
        R_PORI2XYZ = self.Rot_PORI_to_WorldXYZ2(ypr)

        # Rover orientation with respective to world frame
        # Rover XYZ frame {X- Anemometer rod , Y - Anemorod middle, Z - UP}
        R_ROVER2XYZ = R_PORI2XYZ @ R_PORI2ROVER.T

        return R_ROVER2XYZ

    def Rot_PORI_to_WorldXYZ(self, ypr):
        #yaw_error = self.yaw_init - self.base_yaw_init

        # World XYZ frame {X- East, Y - North, Z - UP}
        # World NWU frame {X- North, (-)Y - West, Z - UP}
        # World NWU with respective to World fixed XYZ frame = NWU2XYZ
        R_NWU2XYZ = np.array([[0.0,-1.0, 0.0],
                              [1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0]])
        
        # Physical Orientation of Rover IMU (PORI) measurements with repective to World fixed NED frame =  PORI2NED
        # Vector-nav IMU is used (Absolute mode NED (measure YPR angles deg respective to magnetic North pole))
        # R_PORI2NWU = Rmat.from_euler('zyx', (ypr[0],ypr[1],-ypr[2]), degrees=True).as_matrix() 
        R_PORI2NWU = self.ypr_to_R(ypr)

        # Physical Orientation of Rover IMU (PORI) measurements with repective to World fixed XYZ frame =  PORI2XYZ
        R_PORI2XYZ = R_NWU2XYZ @ R_PORI2NWU

        return R_PORI2XYZ

    def Rot_ROVER_to_WorldXYZ(self, ypr):
        # Physical Orientation of Rover IMU(PORI) with respective to Ship (Printed xyz-axes on IMU orientation respective to ship) = PORI2SHIP
        # {X_PORI-Starbord, Y_PORI-Bow, Z_PORI-Down}
        R_PORI2ROVER = np.array([[0.0, 1.0, 0.0],
                                 [1.0, 0.0, 0.0],
                                 [0.0, 0.0,-1.0]]) 
        
        R_PORI2XYZ = self.Rot_PORI_to_WorldXYZ(ypr)

        # Rover orientation with respective to world frame
        # Rover XYZ frame {X- Anemometer rod , Y - Anemorod middle, Z - UP}
        R_ROVER2XYZ = R_PORI2XYZ @ R_PORI2ROVER.T

        return R_ROVER2XYZ

    def Rot_ROVER_CAMERA_to_WorldXYZ(self, ypr):
        # Rover orientation with respective to world frame
        # Rover XYZ frame {X- Anemometer rod , Y - Anemorod middle, Z - UP}
        R_ROVER2XYZ = self.Rot_ROVER_to_WorldXYZ2(ypr)
        
        R_CAMERA2RoverXYZ = np.array([[1.0, 0.0, 0.0],
                                      [0.0, 0.0,-1.0],
                                      [0.0, 1.0, 0.0]]) 
        
        theta = -self.CameraInclineAngle * np.pi/180
        R_incline = np.array([[1.0,           0.0,           0.0],
                              [0.0, np.cos(theta),-np.sin(theta)],
                              [0.0, np.sin(theta), np.cos(theta)]]) 
        
        phi = -self.yaw_init_corr * np.pi/180
        R_yaw = np.array([[np.cos(phi),0.0,  np.sin(phi)],
                          [0.0, 1.0, 0.0],
                         [-np.sin(phi), 0.0, np.cos(phi)]]) 
        
        

        # Rover orientation with respective to world frame
        # Rover XYZ frame {X- Anemometer rod , Y - Anemorod middle, Z - UP}
        R_CAMERA2XYZ = R_ROVER2XYZ @ R_CAMERA2RoverXYZ @ R_yaw @ R_incline

        return R_CAMERA2XYZ

    def RoverGPSPos2Ship(self, RTK_NED, base_ypr):
        base_ypr[2] = base_ypr[2]
        # T_BA2SO = BaseAntena position from Ship origin with respective to Ship frame 
        T_BA2SO = np.expand_dims(self.T_BASO_ship, axis=1)
        T_BA2ShipOrigin = T_BA2SO.copy()
        T_BA2ShipOrigin[0] = -T_BA2SO[0]
        T_BA2ShipOrigin[2] = T_BA2SO[2] + self.CamPos[2]
        # P_Ro2SO = Rover position from Ship origin
        # CamH = camera height from Ship orgin (from landing pad)
        # RTK_XYZ (w) = RTK_NED
        RTK_NED = np.expand_dims(RTK_NED, axis=1)
        RTK_XYZ = RTK_NED.copy()
        RTK_XYZ[1] = -RTK_NED[1]
        RTK_XYZ[2] = -RTK_NED[2]
        P_wXYZ = RTK_XYZ
        # Ship orientation with respective to world frame
        R_SHIP2XYZ = self.Rot_SHIP_to_WorldXYZ(base_ypr)
        # World XYZ orientation with respective to Ship frame
        R_XYZ2SHIP = R_SHIP2XYZ.T

        # Rover Position from ship origin with respective to ship frame
        P_RP2SO = R_XYZ2SHIP @ P_wXYZ + T_BA2ShipOrigin

        return np.around(P_RP2SO, decimals=6)

    def RoverGPSPos2World(self, RTK_NED, base_ypr):
        # Rover Position from ship origin with respective to ship frame
        P_Ship = self.RoverGPSPos2Ship(RTK_NED, base_ypr)
        # Ship orientation with respective to world frame
        R_SHIP2XYZ = self.Rot_SHIP_to_WorldXYZ(base_ypr)

        # Rover Position from World origin with respective to World frame
        P_RP2WO = R_SHIP2XYZ @ P_Ship 

        return np.around(P_RP2WO, decimals=3)
        
    def Camera_Pose(self, RTK_NED, base_ypr, ypr):
        R_CAMERA2XYZ = self.Rot_ROVER_CAMERA_to_WorldXYZ(ypr)
        T_w = self.RoverGPSPos2World(RTK_NED, base_ypr) # can be use ship
        #T_s = RoverGPSPos2Ship(RTK_NED, base_ypr, T_BA2SO, Z_CS)
        H = np.concatenate((R_CAMERA2XYZ, T_w), axis=1)
        
        return H
    
    def ypr2Rot(self, ypr, degrees = True):
        if degrees:
            ypr = ypr * np.pi / 180
        y = ypr[0]
        p = ypr[1]
        r = ypr[2]
        Rz = np.array([[np.cos(y), -np.sin(y), 0],
                    [np.sin(y),  np.cos(y), 0],
                    [        0,          0, 1]])  # yaw
        Ry = np.array([[ np.cos(p),  0, np.sin(p)],
                    [         0,  1,         0],
                    [-np.sin(p),  0, np.cos(p)]]) # pitch
        Rx = np.array([[1,        0,          0],
                    [0, np.cos(r),-np.sin(r)],
                    [0, np.sin(r), np.cos(r)]])  # roll
        return Rz@Ry@Rx

    def ypr_to_R(self, ypr, degrees = True):
        # http://msl.cs.uiuc.edu/planning/node102.html
        if degrees:
            ypr = ypr * np.pi / 180
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(ypr[0]) * np.cos(ypr[1])
        R[0, 1] = np.cos(ypr[0]) * np.sin(ypr[2]) * np.sin(ypr[1]) - np.cos(ypr[2]) * np.sin(ypr[0])
        R[0, 2] = np.sin(ypr[0]) * np.sin(ypr[2]) + np.cos(ypr[0]) * np.cos(ypr[2]) * np.sin(ypr[1])
        R[1, 0] = np.cos(ypr[1]) * np.sin(ypr[0])
        R[1, 1] = np.cos(ypr[0]) * np.cos(ypr[2]) + np.sin(ypr[0]) * np.sin(ypr[2]) * np.sin(ypr[1])
        R[1, 2] = np.cos(ypr[2]) * np.sin(ypr[0]) * np.sin(ypr[1]) - np.cos(ypr[0]) * np.sin(ypr[2])
        R[2, 0] = - np.sin(ypr[1])
        R[2, 1] = np.cos(ypr[1]) * np.sin(ypr[2])
        R[2, 2] = np.cos(ypr[2]) * np.cos(ypr[1])

        return R

    def rtk_x2rtk_X(self):  
        rtk_X = self.rtk_x.copy()
        for i in range(self.rtk_x.shape[1]):
            R_NED2SHIP = self.Rot_NED_to_SHIP(self.Base_YPR[:,i])
            P_ship = R_NED2SHIP @  np.expand_dims(self.rtk_x[:,i], axis=1)
            rtk_X[:,i] = P_ship.T[0] + self.T_BASO_ship + self.CamPos
        return rtk_X
    
    def rtk_x2rtk_X_cam(self):  
        rtk_X_cam = self.rtk_x_cam.copy()
        try:
            for i in range(self.rtk_x_cam.shape[1]):
                R_NED2SHIP = self.Rot_NED_to_SHIP(self.Base_YPR_cam[:,i])
                P_ship = R_NED2SHIP @  np.expand_dims(self.rtk_x_cam[:,i], axis=1)
                rtk_X_cam[:,i] = P_ship.T[0] + self.T_BASO_ship + self.CamPos
        except:
            pass
        return rtk_X_cam

    def rtk_x2rtk_X_Y(self):  
        rtk_X = self.rtk_x.copy()
        for i in range(self.rtk_x.shape[1]):
            R_NED2SHIP = self.Rot_NED_to_SHIP_Y(self.Base_YPR[:,i])
            P_ship = R_NED2SHIP @  np.expand_dims(self.rtk_x[:,i], axis=1)
            rtk_X[:,i] = P_ship.T[0] + self.T_BASO_ship + self.CamPos
        return rtk_X
    
    def rtk_x2rtk_X_cam_Y(self):  
        rtk_X_cam = self.rtk_x_cam.copy()
        try:
            for i in range(self.rtk_x_cam.shape[1]):
                R_NED2SHIP = self.Rot_NED_to_SHIP_Y(self.Base_YPR_cam[:,i])
                P_ship = R_NED2SHIP @  np.expand_dims(self.rtk_x_cam[:,i], axis=1)
                rtk_X_cam[:,i] = P_ship.T[0] + self.T_BASO_ship + self.CamPos
        except:
            pass
        return rtk_X_cam

    
    def base_yaw_from_llh(self, n):
        ENU = np.zeros((3,self.base_llh[0, :].shape[0]))
        ENU[0,:],ENU[1,:],ENU[2,:] = pm.geodetic2enu(self.base_llh[0, :], self.base_llh[1, :], self.base_llh[2, :], self.base_llh[0, 0], self.base_llh[1, 0], self.base_llh[2, 0])
        return np.arctan2(-ENU[0,0]+ENU[0,n], ENU[1,n]-ENU[1,0]) * 180 / np.pi
    
    def base_yaw_correction(self, base_ypr, YawBase, YawBaseCorrecton = 0):
        try:
            base_ypr = base_ypr.copy()
            base_ypr[0,:] = base_ypr[0,:] - base_ypr[0,0] + YawBase + YawBaseCorrecton
        except:
            base_ypr = np.array([0,0,0])
        return base_ypr
    
    def Rover_yaw_from_RTK_v(self):
        return np.arctan2(self.rtk_v[1, 0],self.rtk_v[0, 0]) * 180 / np.pi

    def rover_yaw_correction(self, ypr, YawRover):
        ypr = ypr.copy()
        try:
            ypr[0,:] = ypr[0,:] - ypr[0,0] + YawRover
        except:
            pass
        return ypr
    
    def rover_yaw_correction_aligned(self, ypr):
        # assume rover axis align with base 
        # initially relative yaw becomes zero
        ypr = ypr.copy()
        try:
            ypr[0,:] = ypr[0,:] - ypr[0,0]
        except:
            pass
        return ypr
    
    def maxRTKvalues(self):
        x = np.max(np.abs(self.rtk_x[0,:]))
        y = np.max(np.abs(self.rtk_x[1,:]))
        z = np.max(np.abs(self.rtk_x[2,:]))
        return np.max(np.array([x, y, z]))
