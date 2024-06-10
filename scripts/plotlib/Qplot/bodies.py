import numpy as np
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from .arrow3d import arrow3Dreletivefunc
from .arrow3d import arrow3Dfunc
from .annotation3D import annotate3Dfunc

lbl_count = 0

setattr(Axes3D, 'arrow3Dr', arrow3Dreletivefunc)
setattr(Axes3D, 'arrow3D', arrow3Dfunc)
setattr(Axes3D, 'annotate3D', annotate3Dfunc)


def plot3D_WorldArrowAxesRelative(ax, P = [0, 0, 0], axesName = ['O','x','y','z']):
    
    
    ax.arrow3Dr(P,[1,0,0],mutation_scale=10,ec ='red',fc='red')
    ax.arrow3Dr(P,[0,1,0],mutation_scale=10,ec ='green',fc='green')
    ax.arrow3Dr(P,[0,0,1],mutation_scale=10,ec ='blue',fc='blue')
    ax.scatter(P[0], P[1], P[2], color="g", s=50)

    ax.annotate3D(axesName[0], (P[0], P[1], P[2]), xytext=(-5, -12), textcoords='offset points')
    ax.annotate3D(axesName[1], (P[0]+1, P[1], P[2]), xytext=(-6, -8), textcoords='offset points')
    ax.annotate3D(axesName[2], (P[0], P[1]+1, P[2]), xytext=(0, 0), textcoords='offset points')
    ax.annotate3D(axesName[3], (P[0], P[1], P[2]+1), xytext=(0, 0), textcoords='offset points')

def plot3D_WorldArrowAxes(ax,scale=1, P = [0, 0, 0], axesName = ['$O_w$','$X_w$','$Y_w$','$Z_w$']):
    
    ax.arrow3D(P,[P[0]+scale,P[1],P[2]],mutation_scale=5,ec ='red',fc='red')
    ax.arrow3D(P,[P[0],P[1]+scale,P[2]],mutation_scale=5,ec ='green',fc='green')
    ax.arrow3D(P,[P[0],P[1],P[2]+scale],mutation_scale=5,ec ='blue',fc='blue')
    ax.scatter(P[0], P[1], P[2], color="g", s=30)

    ax.annotate3D(axesName[0], (P[0], P[1], P[2]), xytext=(-5, -12), textcoords='offset points')
    ax.annotate3D(axesName[1], (P[0]+scale, P[1], P[2]), xytext=(-6, -8), textcoords='offset points')
    ax.annotate3D(axesName[2], (P[0], P[1]+scale, P[2]), xytext=(0, 0), textcoords='offset points')
    ax.annotate3D(axesName[3], (P[0], P[1], P[2]+scale), xytext=(0, 0), textcoords='offset points')
    
def plot3D_NED_axes(ax,scale=1, P = [0, 0, 0], axesName = ['$0$','$N$','$E$','$D$']):
    
    ax.arrow3D(P,[P[0]+scale,P[1],P[2]],mutation_scale=5,ec ='red',fc='red')
    ax.arrow3D(P,[P[0],P[1]+scale,P[2]],mutation_scale=5,ec ='green',fc='green')
    ax.arrow3D(P,[P[0],P[1],-P[2]+scale],mutation_scale=5,ec ='blue',fc='blue')
    ax.scatter(P[0], P[1], P[2], color="g", s=30)

    ax.annotate3D(axesName[0], (P[0], P[1], P[2]), xytext=(-5, -12), textcoords='offset points')
    ax.annotate3D(axesName[1], (P[0]+scale, P[1], P[2]), xytext=(-6, -8), textcoords='offset points')
    ax.annotate3D(axesName[2], (P[0], P[1]+scale, P[2]), xytext=(0, 0), textcoords='offset points')
    ax.annotate3D(axesName[3], (P[0], P[1], P[2]+scale), xytext=(0, 0), textcoords='offset points')


def set_axeslim(ax, P=[0,0,0], Q=[0,0,0]):    
    ax.set_xlim(min(P[0],Q[0])-1.25,max(P[0],Q[0])+2.25)
    ax.set_ylim(min(P[1],Q[1])-1.25,max(P[1],Q[1])+2.25)
    ax.set_zlim(min(P[2],Q[2])-1.25,max(P[2],Q[2])+2.25)

def plot3D_LineWorldAxes(ax, P = [0, 0, 0]):
    ax.plot([P[0],P[0]+1], [P[1],P[1]], [P[2],P[2]],color='blue', linewidth=1)
    ax.plot([P[0],P[0]], [P[1],P[1]+1], [P[2],P[2]],color='green', linewidth=1)
    ax.plot([P[0],P[0]], [P[1],P[1]], [P[2],P[2]+1],color='red', linewidth=1)
    ax.scatter(P[0], P[1], P[2], color="g", s=50)

def plotCtoW(*args, **kwargs):
    ax = args[0]
    pointer = False
    try:
        H=kwargs['H']
        S=kwargs['scale']
        pointer=kwargs['pointer'] 
#        R = H[0:3,0:3]
#        T = H[0:3,3]
    except:
        try:
            R=kwargs['R']
            T=kwargs['T']
            S=kwargs['scale']
            pointer=kwargs['pointer'] 
        except:
            pass
    try:
          axesName = kwargs['axes']
    except:
        axesName = ['O','x','y','z']

    P_c = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])*S
    P = P_c@R.T + np.tile(T,(P_c.shape[0],1)) 
        
    ax.scatter(P[0,0], P[0,1], P[0,2], color="black", s=20)
    ax.arrow3D(P[0,:], P[1,:],mutation_scale=5,ec ='red',fc='red')
    ax.arrow3D(P[0,:], P[2,:],mutation_scale=5,ec ='green',fc='green')
    ax.arrow3D(P[0,:], P[3,:],mutation_scale=5,ec ='blue',fc='blue')
    
    ax.annotate3D(axesName[0], (P[0,:]), xytext=(-5, -12), textcoords='offset points')
    ax.annotate3D(axesName[1], (P[1,:]), xytext=(-6, -8), textcoords='offset points')
    ax.annotate3D(axesName[2], (P[2,:]), xytext=(0, 0), textcoords='offset points')
    ax.annotate3D(axesName[3], (P[3,:]), xytext=(0, 0), textcoords='offset points')
    
    if pointer:
        ax.arrow3D([0, 0, 0],P[0,:],mutation_scale=10,arrowstyle="-|>",linestyle='dashed')
        
def plotArrow3D(ax, **kwargs):
    global lbl_count
    _P=np.array([0, 0, 0])
    _Q=np.array([1, 1, 1])
    color='black'
    size = 1
    Label = "None"
    try:
        _P=kwargs['P']
        _Q=kwargs['Q']
        color=kwargs['color'] 
        size=kwargs['size'] 
        Label=kwargs['label'] 
    except:
        pass
        
    if lbl_count < 2 and (Label=='Anemometer 1' or Label=='Anemometer 2'):
        ax.arrow3D(_P,_Q,mutation_scale=size,fc=color,ec=color, arrowstyle="->", label=Label)
        lbl_count += 1
    elif lbl_count < 2 and Label=='Bow Anemometer':
        ax.arrow3D(_P,_Q,mutation_scale=size,fc=color,ec=color, arrowstyle="->", label=Label)
        lbl_count += 2
    else:
        ax.arrow3D(_P,_Q,mutation_scale=size,fc=color,ec=color, arrowstyle="->")
    #ax.arrow3D(_P,_Q,mutation_scale=size,arrowstyle="-|>",linestyle='dashed',ec=color,fc=color)
    
# def plotArrow3Dr(ax, **kwargs):
#     _P=np.array([0, 0, 0])
#     _Q=np.array([1, 1, 1])
#     color='black'
#     size = 5
#     Label = "None"
#     try:
#         _P=kwargs['P']
#         _Q=kwargs['Q']
#         color=kwargs['color'] 
#         size=kwargs['size'] 
#         Label=kwargs['label'] 
#     except:
#         pass
    

#     ax.arrow3Dr(_P,_Q,mutation_scale=size,ec=color,fc=color, label=Label)
    

def plotArrow3Dr(ax, **kwargs):
    _P=np.array([0, 0, 0])
    _Q=np.array([1, 1, 1])
    color='black'
    size = 5
    Label = "None"
    try:
        _P=kwargs['P']
        _Q=kwargs['Q']
        color=kwargs['color'] 
        size=kwargs['size'] 
        Label=kwargs['label'] 
    except:
        pass
    ax.arrow3Dr(_P,_Q,mutation_scale=size,ec=color,fc=color, label=Label, arrowstyle="->")
    
    
def plot3Dpoint(ax, **kwargs):
    P=[0, 0, 0]
    color='black'
    size = 20
    try:
        P=kwargs['P']
        color=kwargs['color'] 
        size = kwargs['size']
    except:
        pass

    ax.scatter(P[0], P[1], P[2], color="black", s=size)

    
def plotCamera(*args, **kwargs):
    ax = args[0]
    S = 1
    try:
        _H=kwargs['H']
        _R = _H[0:3,0:3]
        _T = _H[0:3,3]
        S=kwargs['scale']
    except:
        try:
            _R=kwargs['R']
            _T=kwargs['T']
            S=kwargs['scale']
        except:
            pass
    
    b = 0.108*3*S
    a = 0.192*3*S
    c = -0.8*S
    
    Cam_B = np.array([[0,0,0],[a/2,b/2,c],[-a/2,b/2,c],[-a/2,-b/2,c],[a/2,-b/2,c],
                      [a/2,b/2,c],[0.09,b/2,c],[0,b/2+0.08,c],[-0.09,b/2,c],[-a/2,b/2,c],
                      [0,0,0],[-a/2,b/2,c],[0,0,0],[-a/2,-b/2,c],[0,0,0],[a/2,-b/2,c]])
    Cam_A = Cam_B@_R.T + np.tile(_T,(Cam_B.shape[0],1))
    Cam_A = Cam_A.T
    ax.plot(Cam_A[0], Cam_A[1], Cam_A[2],color='dimgray', linewidth=1)
    
    
def transform_vertices_groups(vertices_groups, **kwargs):
    Rot=np.eye(3)
    Tr=np.array([0,0,0])
    S = 1
    try:
        Rot = kwargs['R']
        Tr = kwargs['T']
        S = kwargs['scale']
    except:
        try:
            Tr = kwargs['T']
            Rot = kwargs['R']
            S = kwargs['scale']
        except:
            try:
                S = kwargs['scale']
            except:
                pass

    for i in range(len(vertices_groups)):
        vertices_group_arr = np.asarray(vertices_groups[i])*S
        transformed_vertices_group_arr = Rot @ vertices_group_arr.T + np.tile(Tr,(vertices_group_arr.shape[0],1)).T
        transformed_vertices_group_arr = transformed_vertices_group_arr
        if i == 0:
            tr_vertices_groups = [transformed_vertices_group_arr.T.tolist()]
        else:
            tr_vertices_groups.extend([transformed_vertices_group_arr.T.tolist()])

    return tr_vertices_groups

def arc2d_vertices(r,n=24,theta=360):
    N = 360/n
    arr = []
    for i in range(int(theta/N)):
        arr = np.append(arr,np.array([r*np.cos(i*N*np.pi/180),r*np.sin(i*N*np.pi/180),0]))
    arr = arr.reshape(int(theta/N),3)
    tr_vertices_groups = [arr.tolist()]
    return tr_vertices_groups

def transform_wind(W, TW, size=0.5, **kwargs):
    Rot=np.eye(3)
    Rotw = np.array([[0, -1, 0],[1, 0, 0],[0, 0, 1]])
    Tr=np.array([0,0,0])
    
    S = 1
    try:
        Rot = kwargs['R']
        Tr = kwargs['T']
        S = kwargs['scale']
    except:
        try:
            Tr = kwargs['T']
            Rot = kwargs['R']
            S = kwargs['scale']
        except:
            try:
                S = kwargs['scale']
            except:
                pass
    Tr = np.expand_dims(Tr, axis=1) #axis 1 [[0][0][0]] #axis 1 [[0 0 0]]
    Tw = np.expand_dims(TW, axis=1)
    W = np.expand_dims(W, axis=1)
 
    T = Tr + Rot @ Tw*S

    Pw = Rot @ Rotw @ W * size +  T

    return Pw.T[0]

