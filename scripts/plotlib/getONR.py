import os
import numpy as np
import pandas as pd
from configparser import ConfigParser, ExtendedInterpolation
import json
from .ONR import ONR

def calcONRfromData(DirGPS):
    for Testfile in os.listdir(DirGPS):
        if Testfile.endswith(".csv"):
            print(Testfile)
            file_name = DirGPS+Testfile

    for Testfile in os.listdir(DirGPS):        
        if Testfile.endswith(".ini"):
            print(Testfile)
            configfile = DirGPS+Testfile
            config = ConfigParser()
            config.read(configfile)

            cam_Incline_Angle = config.getfloat('initial', 'cam_Incline_Angle')
            base_yaw_init_correct = config.getfloat('initial', 'base_yaw_init_correct') 
            try:
                yaw_init_corr = config.getfloat('initial', 'yaw_init_corr')
            except:
                yaw_init_corr = 0.0
            T_BASO_ship = np.array(json.loads(config.get('initial', 'T_BASO_ship')))
            CamPos = np.array(json.loads(config.get('initial', 'CamPos')))
            skip = config.getint('initial', 'skip')
            end_skip = config.getint('initial', 'end_skip')
            break
        else:
            cam_Incline_Angle = 1.4
            base_yaw_init_correct = -3.1
            T_BASO_ship = [0.150, 2.68,1.5]
            CamPos = [0.0, 0, 0.23]
            skip = 1
            end_skip = 1

 
    #df = pd.read_csv(file_name)
    df = pd.read_csv(file_name, header=0, skipinitialspace=True)
    cols = ['ypr_0', 'ypr_1', 'ypr_2', 'a_0', 'a_1', 'a_2', 'W_0', 'W_1', 'W_2']

    # Outlier rejection
    Q1 = df[cols].quantile(0.05)
    Q3 = df[cols].quantile(0.95)
    IQR = Q3 - Q1

    df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
    onr = ONR(df)
    skip = 2
    end_skip = 1
    m=onr.rtk_x[0][onr.rtk_x[0]==0].shape[0]+skip
    df = df.iloc[m:-end_skip]
    onr = ONR(df, base_yaw_init_correct= base_yaw_init_correct, cam_Incline_Angle = cam_Incline_Angle, T_BASO_ship = T_BASO_ship, CamPos = CamPos, yaw_init_corr= yaw_init_corr)

    return onr
