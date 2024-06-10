import numpy as np
from .np_projectiony import project3Dto2D

cat_names = ['dog house','dog house left','dog house right','dog house center','house','house long','whole ship',"super structure",'ship stern','landing pad',"Mast",'pelican case']

def cat_names_to_id(name="dog house"):
    return cat_names.index(name)+1

def cat_points(cat_id = 1):
    if cat_id == 0:
        cat_id = 1
    if cat_id == cat_names_to_id("pelican case"): #10
        x1 = 0.543
        x2 = 0.12
        y1 = 2.37
        y2 = 2.77
        z1 = 0
        z2 = 0.14

        P = np.array([[x1, y1, z1],[x1, y1, z2],[x2, y1, z2],[x2, y1, z1],[x1, y2, z1],[x1, y2, z2],[x2, y2, z2],[x2, y2, z1]])

        return P

    if cat_id == cat_names_to_id("dog house"): #1
        x1 = 2.11022 
        x2 = 1.94844
        y1 = 2.78
        y2 = 3.80835  #5.5
        z1 = 0
        z2 = 1.6134

        P = np.array([[x1, y1, z1],[x2, y1, z2],[-x2, y1, z2],[-x1, y1, z1],[x1, y2, z1],[x2, y2, z2],[-x2, y2, z2],[-x1, y2, z1]])

        return P

    if cat_id == cat_names_to_id("dog house left"): #2
        P = np.array([[0, 2.7533, 0],[0, 2.7533, 1.6133],[-1.94844, 2.7533, 1.55987],[-2.11022, 2.7533, 0],
                      [0, 3.82, 0],[0, 3.82, 1.6133],[-1.94844 , 3.82, 1.55987],[-2.11022, 3.82, 0]])
        return P
        
    if cat_id == cat_names_to_id("dog house right"): #3
        P = np.array([[2.11022, 2.7533, 0],[1.94844, 2.7533, 1.6133],[0, 2.7533, 1.55987],[0, 2.7533, 0],
                      [2.11022, 3.82, 0],[1.94844, 3.82, 1.6133],[0, 3.82, 1.55987],[0, 3.82, 0]])

        return P
        
    if cat_id == cat_names_to_id("dog house center"): #4
        x1 = 0.974595
        x2 = 0.974595
        y1 = 2.78
        y2 = 3.80835  
        z1 = 0
        z2 = 1.60133 

        P = np.array([[x1, y1, z1],[x2, y1, z2],[-x2, y1, z2],[-x1, y1, z1],[x1, y2, z1],[x2, y2, z2],[-x2, y2, z2],[-x1, y2, z1]])

        return P
        
    if cat_id == cat_names_to_id("Mast"): #5
        x1 = 0.152
        x2 = 0.152
        y1 = 3.34081
        y2 = 3.77873
        z1 = 1.62751 
        z2 = 4.52485 

        P = np.array([[x1, y1, z1],[x1, y1, z2],[-x1, y1, z2],[-x1, y1, z1],[x2, y2, z1],[x2, y2, z2],[-x2, y2, z2],[-x2, y2, z1]])

        return P
        
    if cat_id == cat_names_to_id("house"): #5
        x1 = 2.11022 
        x2 = 1.94844
        y1 = 2.78
        y2 = 6.86995  
        z1 = 0
        z2 = 1.6134

        P = np.array([[x1, y1, z1],[x2, y1, z2],[-x2, y1, z2],[-x1, y1, z1],[x1, y2, z1],[x2, y2, z2],[-x2, y2, z2],[-x1, y2, z1]])

        return P
        
    if cat_id == cat_names_to_id("house long"): #6
        x1 = 2.11022 
        x2 = 1.94844
        y1 = 2.78
        y2 = 11.3135   
        z1 = 0
        z2 = 1.6134

        P = np.array([[x1, y1, z1],[x2, y1, z2],[-x2, y1, z2],[-x1, y1, z1],[x1, y2, z1],[x2, y2, z2],[-x2, y2, z2],[-x1, y2, z1]])

        return P

    if cat_id == cat_names_to_id("whole ship"): #7
        x1 = 3.33433 
        x2 = -3.33433 
        y1 = -2.92091
        y2 = 19.8255
        z1 = -2.27976 
        z2 = 2.82816

        P = np.array([[x1, y1, z1],[x1, y1, z2],[x2, y1, z2],[x2, y1, z1],[x1, y2, z1],[x1, y2, z2],[x2, y2, z2],[x2, y2, z1]])

        return P
        
    if cat_id == cat_names_to_id("super structure"): #9
        x1 = 3.33581
        x2 = 3.33581 
        y1 = 2.78005 
        y2 = 19.8254 
        z1 = 4.14937
        z2 = 0

        P = np.array([[x1, y1, z1],[x1, y1, z2],[-x1, y1, z2],[-x1, y1, z1],[x2, y2, z1],[x2, y2, z2],[-x2, y2, z2],[-x2, y2, z1]])

        return P

        
    if cat_id == cat_names_to_id("landing pad"): #8
        x1 = 3.138
        x2 = 3.334 
        y1 = -2.920
        y2 = 2.78
        z1 = 0
        z2 = -0.810107

        P = np.array([[x1, y1, z1],[x1, y1, z2],[-x1, y1, z2],[-x1, y1, z1],[x2, y2, z1],[x2, y2, z2],[-x2, y2, z2],[-x2, y2, z1]])

        return P


    if cat_id == cat_names_to_id("ship stern"): #9
        x1 = 3.138
        x2 = 3.334 
        y1 = -2.920
        y2 = 2.78
        z1 = 0
        z2 = -1.89534

        P = np.array([[x1, y1, z1],[x1, y1, z2],[-x1, y1, z2],[-x1, y1, z1],[x2, y2, z1],[x2, y2, z2],[-x2, y2, z2],[-x2, y2, z1]])

        return P


def knownWorld3dBoxPoints(P, points_only = 0):
    if points_only == 0:
        
        P1 = P[0,:]
        P2 = P[1,:]
        P3 = P[2,:]
        P4 = P[3,:]
        P5 = P[4,:]
        P6 = P[5,:]
        P7 = P[6,:]
        P8 = P[7,:]

     
        #Pw_C = np.array([P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12,P13,P14,P15,P16,P17,P18]) # corner vertices
        Pw_C = np.array([P1,P2,P3,P4,P5,P6,P7,P8])                   
        #Point start and end | interpolated points
        P_se = np.array([[P1,P2],[P2,P3],[P3,P4],[P4,P1],
                            [P1,P5],[P2,P6],[P3,P7],[P4,P8],
                            [P5,P6],[P6,P7],[P7,P8],[P8,P5]])
                    
        b = 2/3
        B = np.array([[b, 1-b],[1-b, b]]) # B_qx2 = interpolation matrix
        l = P_se.shape[0]
        #l=0 #<------------------------tmp
        if l>0:
            for i in range(l):
                pw_I = B@P_se[i]
                if i == 0:
                    Pw_I = pw_I
                else:
                    Pw_I = np.concatenate((Pw_I,pw_I), axis = 0)
        # ones = np.ones((Pw.shape[0],1))
        #Pw = np.concatenate((Pw_C,Pw_I), axis = 0)
        else:
            Pw_I = 0
            Pw = Pw_C
            Pw_axis = np.array([[0,0,0],[0,0,1],[0,0,0],[0,1,0],[0,0,0],[1,0,0]])
            Pw_shifted_axis = np.array([[0,2.325,0],[0,2.325,1],[0,2.325,0],[0,2.87,0],[0,2.325,0],[1,2.325,0]])

            return Pw, Pw_C, Pw_I, Pw_axis, Pw_shifted_axis

    

    Pw_axis = np.array([[0,0,0],[0,0,1],[0,0,0],[0,1,0],[0,0,0],[1,0,0]])

    Pw_shifted_axis = np.array([[0,2.325,0],[0,2.325,1],[0,2.325,0],[0,2.87,0],[0,2.325,0],[1,2.325,0]])

    #Pw_window = np.array([[-0.405,2.856,1.325],[0.295,2.856,1.325],[0.295,2.856,0.68],[-0.405,2.856,0.68]])

    Pw = np.concatenate((Pw_C,Pw_I), axis = 0)
    
    return Pw

def projectedKnown3dBoxPoints(P, Hc,K, points_only = 0):
    Pw, Pw_C, Pw_I, Pw_axis, Pw_shifted_axis = knownWorld3dBoxPoints(P, points_only = points_only)
    Pc_axis = project3Dto2D(Pw_axis,Hc,K)
    Pc_C = project3Dto2D(Pw_C,Hc,K)
    if Pw_I.shape[0] > 0:
        Pc_I = project3Dto2D(Pw_I,Hc,K)
    else:
        Pc_I = 0
    #Pc_box = project3Dto2D(Pw_box,Hc,K)
    Pc = project3Dto2D(Pw,Hc,K)

    Pc_shifted_axis = project3Dto2D(Pw_shifted_axis,Hc,K)
    return Pc, Pc_C, Pc_I, Pc_axis, Pc_shifted_axis

def pointsOnALine(P,i):
    if i < 3:
        P_L = np.concatenate(([P[i]],[P[2*i+8]],[P[2*i+9]],[P[i+1]]), axis = 0)
        P_c = np.concatenate(([P[i]],[P[i+1]]), axis = 0)
        P_I = np.concatenate(([P[2*i+8]],[P[2*i+9]]), axis = 0)
    if i == 3:
        P_L = np.concatenate(([P[i]],[P[2*i+8]],[P[2*i+9]],[P[i-3]]), axis = 0)
        P_c = np.concatenate(([P[i]],[P[i-3]]), axis = 0)
        P_I = np.concatenate(([P[2*i+8]],[P[2*i+9]]), axis = 0)
    if i > 3 and i < 8:
        P_L = np.concatenate(([P[i-4]],[P[2*i+8]],[P[2*i+9]],[P[i]]), axis = 0)
        P_c = np.concatenate(([P[i-4]],[P[i]]), axis = 0)
        P_I = np.concatenate(([P[2*i+8]],[P[2*i+9]]), axis = 0)
    if i >= 8 and i < 11:
        P_L = np.concatenate(([P[i-4]],[P[2*i+8]],[P[2*i+9]],[P[i-3]]), axis = 0)
        P_c = np.concatenate(([P[i-4]],[P[i-3]]), axis = 0)
        P_I = np.concatenate(([P[2*i+8]],[P[2*i+9]]), axis = 0)
    if i == 11:
        P_L = np.concatenate(([P[i-4]],[P[2*i+8]],[P[2*i+9]],[P[i-7]]), axis = 0)
        P_c = np.concatenate(([P[i-4]],[P[i-7]]), axis = 0)
        P_I = np.concatenate(([P[2*i+8]],[P[2*i+9]]), axis = 0)
    return P_L, P_c, P_I
