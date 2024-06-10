import numpy as np

def Hw2Hc4x4(Hw):
    Hc = np.zeros((4,4))
    Hc[0:3,0:3] = Hw[0:3,0:3].T
    Hc[0:3,3] = -Hw[0:3,0:3].T @ Hw[0:3,3]
    Hc[3,3] = 1
    return Hc

def Hc2Hw4x4(Hc):
    Hw = np.zeros((4,4))
    Hw[0:3,0:3] = Hc[0:3,0:3].T
    Hw[0:3,3] = -Hc[0:3,0:3].T @ Hc[0:3,3]
    Hw[3,3] = 1
    return Hw

def Hw2Hc3x4(Hw):
    Hc = np.zeros((3,4))
    Hc[0:3,0:3] = Hw[0:3,0:3].T
    Hc[0:3,3] = -Hw[0:3,0:3].T @ Hw[0:3,3]
    return Hc

def Hc2Hw3x4(Hc):
    Hw = np.zeros((3,4))
    Hw[0:3,0:3] = Hc[0:3,0:3].T
    Hw[0:3,3] = -Hc[0:3,0:3].T @ Hc[0:3,3]
    return Hw