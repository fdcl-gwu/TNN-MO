import numpy as np
from scipy.spatial.transform import Rotation as Rmat
import pandas as pd


def Rot_POBI_to_WorldXYZ(base_ypr):
    # World XYZ frame {X- East, Y - North, Z - UP}
    # World NED frame {X- North, Y - East, Z - Down}
    # World NED with respective to World fixed XYZ frame = NED2XYZ
    R_NED2XYZ = np.array([[0.0, 1.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 0.0,-1.0]])
       
    # Physical Orientation of Base IMU (POBI) measurements with repective to World fixed NED frame =  POBI2NED
    # Vector-nav IMU is used (Absolute mode NED (measure YPR angles deg respective to magnetic North pole))
    R_POBI2NED = Rmat.from_euler('zyx', (-base_ypr[0],-base_ypr[1],base_ypr[2]), degrees=True).as_matrix() # YPR ++- / --+

    # Physical Orientation of Base IMU (POBI) measurements with repective to World fixed XYZ frame =  POBI2XYZ
    R_POBI2XYZ = R_NED2XYZ @ R_POBI2NED

    return R_POBI2XYZ

def Rot_SHIP_to_WorldXYZ(base_ypr):
    # Physical Orientation of Base IMU (POBI) with respective to Ship (Printed xyz-axes on IMU orientation respective to ship) = POBI2SHIP
    # {X_POBI-Bow, Y_POBI-Port Z_POBI-Up}
    R_POBI2SHIP = np.array([[0.0,-1.0,0.0],
                            [1.0, 0.0,0.0],
                            [0.0, 0.0,1.0]])
    
    R_POBI2XYZ = Rot_POBI_to_WorldXYZ(base_ypr)

    # Ship orientation with respective to world frame
    # Ship XYZ frame {X- Starboard, Y - Bow, Z - UP}
    R_SHIP2XYZ = R_POBI2XYZ @ R_POBI2SHIP.T

    return R_SHIP2XYZ


# def SHIP_to_WorldXYZ_bkp(base_ypr):
#     # World XYZ frame {X- East, Y - North, Z - UP}
#     # World NED frame {X- North, Y - East, Z - Down}
#     # World NED with respective to World fixed XYZ frame = NED2XYZ
#     R_NED2XYZ = np.array([[0.0, 1.0, 0.0],
#                           [1.0, 0.0, 0.0],
#                           [0.0, 0.0,-1.0]])
    
#     # Physical Orientation of Base IMU (POBI) with respective to Ship (Printed xyz-axes on IMU orientation respective to ship) = POBI2SHIP
#     # {X_POBI-Bow, Y_POBI-Port Z_POBI-Up}
#     R_POBI2SHIP = np.array([[0.0,-1.0,0.0],
#                             [1.0, 0.0,0.0],
#                             [0.0, 0.0,1.0]])
       
#     # Physical Base IMU Orientation (POBI) measurements with repective to World fixed NED frame =  POBI2NED
#     # Vector-nav IMU is used (Absolute mode NED (measure YPR angles deg respective to magnetic North pole))
#     R_POBI2NED = Rmat.from_euler('zyx', (base_ypr[0],base_ypr[1],-base_ypr[2]), degrees=True).as_matrix() 

#     # Ship orientation with respective to world frame
#     # Ship XYZ frame {X- Starboard, Y - Bow, Z - UP}
#     R_SHIP2XYZ = R_NED2XYZ @ R_POBI2NED @ R_POBI2SHIP.T

#     return R_SHIP2XYZ


def Rot_PORI_to_WorldXYZ(ypr, yaw_init = 0, base_yaw_init = 0):
    yaw_error = yaw_init-base_yaw_init

    # World XYZ frame {X- East, Y - North, Z - UP}
    # World NWU frame {X- North, (-)Y - West, Z - UP}
    # World NWU with respective to World fixed XYZ frame = NWU2XYZ
    R_NWU2XYZ = np.array([[0.0,-1.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0]])
    
    # Physical Orientation of Rover IMU (PORI) measurements with repective to World fixed NED frame =  PORI2NED
    # Vector-nav IMU is used (Absolute mode NED (measure YPR angles deg respective to magnetic North pole))
    R_PORI2NWU = Rmat.from_euler('zyx', (ypr[0]-yaw_error,ypr[1],-ypr[2]), degrees=True).as_matrix() 

    # Physical Orientation of Rover IMU (PORI) measurements with repective to World fixed XYZ frame =  PORI2XYZ
    R_PORI2XYZ = R_NWU2XYZ @ R_PORI2NWU

    return R_PORI2XYZ

def Rot_ROVER_to_WorldXYZ(ypr, yaw_init = 0, base_yaw_init = 0):
    # Physical Orientation of Rover IMU(PORI) with respective to Ship (Printed xyz-axes on IMU orientation respective to ship) = PORI2SHIP
    # {X_PORI-Starbord, Y_PORI-Bow, Z_PORI-Down}
    R_PORI2ROVER = np.array([[0.0, 1.0, 0.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 0.0,-1.0]]) 
    
    R_PORI2XYZ = Rot_PORI_to_WorldXYZ(ypr, yaw_init, base_yaw_init)

    # Rover orientation with respective to world frame
    # Rover XYZ frame {X- Anemometer rod , Y - Anemorod middle, Z - UP}
    R_ROVER2XYZ = R_PORI2XYZ @ R_PORI2ROVER.T

    return R_ROVER2XYZ

def Rot_ROVER_CAMERA_to_WorldXYZ(ypr, yaw_init = 0, base_yaw_init = 0, incline=2):
    # Rover orientation with respective to world frame
    # Rover XYZ frame {X- Anemometer rod , Y - Anemorod middle, Z - UP}
    R_ROVER2XYZ = Rot_ROVER_to_WorldXYZ(ypr, yaw_init, base_yaw_init)
    
    R_CAMERA2RoverXYZ = np.array([[1.0, 0.0, 0.0],
                                  [0.0, 0.0,-1.0],
                                  [0.0, 1.0, 0.0]]) 
    theta = -incline*np.pi/180
    R_incline = np.array([[1.0,           0.0,           0.0],
                          [0.0, np.cos(theta),-np.sin(theta)],
                          [0.0, np.sin(theta), np.cos(theta)]]) 
    # Rover orientation with respective to world frame
    # Rover XYZ frame {X- Anemometer rod , Y - Anemorod middle, Z - UP}
    R_CAMERA2XYZ = R_ROVER2XYZ @ R_CAMERA2RoverXYZ @ R_incline

    return R_CAMERA2XYZ

def RoverGPSPos2Ship(RTK_NED, base_ypr, T_BA2SO = np.array([1.429, 3.49, 2.349]), Z_CS = 0.35):
    base_ypr[2] = base_ypr[2]
    # P_BA2SO = BaseAntena position from Ship origin with respective to Ship frame 
    T_BA2SO = np.expand_dims(T_BA2SO, axis=1)
    T_BA2ShipOrigin = T_BA2SO.copy()
    T_BA2ShipOrigin[0] = -T_BA2SO[0]
    T_BA2ShipOrigin[2] = T_BA2SO[2]+Z_CS
    # P_Ro2SO = Rover position from Ship origin
    # Z_CB = camera height from Ship orgin (landing pad)
    # RTK_XYZ (w) = RTK_NED
    RTK_NED = np.expand_dims(RTK_NED, axis=1)
    RTK_XYZ = RTK_NED.copy()
    RTK_XYZ[1] = -RTK_NED[1]
    RTK_XYZ[2] = -RTK_NED[2]
    P_wXYZ = RTK_XYZ
    # Ship orientation with respective to world frame
    R_SHIP2XYZ = Rot_SHIP_to_WorldXYZ(base_ypr)
    # World XYZ orientation with respective to Ship frame
    R_XYZ2SHIP = R_SHIP2XYZ.T

    # Rover Position from ship origin with respective to ship frame
    P_RP2SO = R_XYZ2SHIP @ P_wXYZ + T_BA2ShipOrigin

    return np.around(P_RP2SO, decimals=6)

def RoverGPSPos2World(RTK_NED, base_ypr, T_BA2SO = np.array([1.429, 3.49, 2.349]), Z_CS = 0.35):
    # Rover Position from ship origin with respective to ship frame
    P_Ship = RoverGPSPos2Ship(RTK_NED, base_ypr, T_BA2SO, Z_CS)
    # Ship orientation with respective to world frame
    R_SHIP2XYZ = Rot_SHIP_to_WorldXYZ(base_ypr)

    # Rover Position from World origin with respective to World frame
    P_RP2WO = R_SHIP2XYZ @ P_Ship 

    return np.around(P_RP2WO, decimals=3)
    
def Camera_Pose(RTK_NED, base_ypr, ypr, yaw_init = 0, base_yaw_init = 0, incline=2, T_BA2SO = np.array([1.429, 3.49, 2.349]), Z_CS = 0.35):
    R_CAMERA2XYZ = Rot_ROVER_CAMERA_to_WorldXYZ(ypr, yaw_init, base_yaw_init, incline)
    T_w = RoverGPSPos2World(RTK_NED, base_ypr, T_BA2SO, Z_CS) # can be use ship
    #T_s = RoverGPSPos2Ship(RTK_NED, base_ypr, T_BA2SO, Z_CS)
    H = np.concatenate((R_CAMERA2XYZ, T_w), axis=1)
    
    return H
    