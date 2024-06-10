import numpy as np
import pandas as pd

class Data:
    def __init__(self, df):
        self.t = self.get(df, 't') / 1000.0
        self.N = len(self.t)
        self.ypr = self.get_3x1(df, 'ypr') / 180.0 * np.pi
        self.R = self.ypr_array_to_R_array(self.ypr)
        self.a = self.get_3x1(df, 'a')
        self.W = self.get_3x1(df, 'W')
        self.rtk_x = self.get_3x1(df, 'rtk_x')
        self.rtk_v = self.get_3x1(df, 'rtk_v')
        self.llh = self.get_3x1(df, 'llh')
        self.gps_status = self.get(df, 'status')
        self.gps_num_sats = self.get(df, 'sats')
        self.ane1 = self.get_3x1(df, 'ane1')
        self.ane2 = self.get_3x1(df, 'ane2')

        self.base_ypr = self.get_3x1(df, 'base_ypr') / 180.0 * np.pi
        self.base_R = self.ypr_array_to_R_array(self.base_ypr)
        self.base_a = self.get_3x1(df, 'base_a')
        self.base_W = self.get_3x1(df, 'base_W')
        self.base_llh = self.get_3x1(df, 'base_llh')
        self.base_gps_status = self.get(df, 'base_status')
        self.base_gps_num_sats = self.get(df, 'base_sats')
        self.base_ane = self.get_3x1(df, 'base_ane')

    
    def get_3x1(self, df, label):

        return np.vstack(
            (
                df['{}_0'.format(label)].values, 
                df['{}_1'.format(label)].values, 
                df['{}_2'.format(label)].values
            )
        )
    

    def get(self, df, label):
        return df[label].values


    def ypr_array_to_R_array(self, ypr_array):
        R = np.zeros((3, 3, self.N))
        for i in range(self.N):
            R[:, :, i] = self.ypr_to_R(ypr_array[:, i])
        
        return R
    
    def ypr_to_R(self, ypr):
  
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
        
    def dt(self):
        print("check")
