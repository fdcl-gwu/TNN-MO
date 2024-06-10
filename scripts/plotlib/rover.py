import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from .Qplot.bodies import *

Rover_vertices1 = [[[-0.0762, -0.0762, -0.0508],[0.0762, -0.0762, -0.0508],[0.0762, -0.0762, 0.0508],[-0.0762, -0.0762, 0.0508]], 
                     [[-0.0762, 0.0762, -0.0508],[0.0762, 0.0762, -0.0508],[0.0762, 0.0762, 0.0508],[-0.0762, 0.0762, 0.0508]],
                     [[-0.0762, -0.0762, -0.0508],[-0.0762, -0.0762, 0.0508],[-0.0762, 0.0762, 0.0508],[-0.0762, 0.0762, -0.0508]],
                     [[0.0762, -0.0762, -0.0508],[0.0762, -0.0762, 0.0508],[0.0762, 0.0762, 0.0508],[0.0762, 0.0762, -0.0508]],
                     [[-0.0762, -0.0762, -0.0508],[0.0762, -0.0762, -0.0508],[0.0762, 0.0762, -0.0508],[-0.0762, 0.0762, -0.0508]],
                     [[-0.0762, -0.0762, 0.0508],[0.0762, -0.0762, 0.0508],[0.0762, 0.0762, 0.0508],[-0.0762, 0.0762, 0.0508]],
                     [[-0.45, 0.08, -0.015],[0.45, 0.08, -0.015],[0.45, 0.08, 0.015],[-0.45, 0.08, 0.015]],
                     [[-0.45, 0.095, -0.015],[0.45, 0.095, -0.015],[0.45, 0.095, 0.015],[-0.45, 0.095, 0.015]],
                     [[-0.45, 0.08, -0.015],[0.45, 0.08, -0.015],[0.45, 0.095, -0.015],[-0.45, 0.095, -0.015]],
                     [[-0.45, 0.08, 0.015],[0.45, 0.08, 0.015],[0.45, 0.095, 0.015],[-0.45, 0.095, 0.015]]]

scale = 0.2
x1 = 0.2 * scale
y1 = 0.05 * scale
z = 0.45 * scale
Anemo_vertices = [[[-x1, y1, 0],[-x1, -y1, 0],[-x1, -y1, z],[-x1, y1, z],[-x1, y1, 0]],
                  [[x1, y1, 0],[x1, -y1, 0],[x1, -y1, z],[x1, y1, z],[x1, y1, 0]],
                  [[-x1, y1, 0],[-x1, -y1, 0],[x1, -y1, 0],[x1, y1, 0],[-x1, y1, 0]],
                  [[-x1, y1, z],[-x1, -y1, z],[x1, -y1, z],[x1, y1, z],[-x1, y1, z]]]
scale = 0.1
x1 = 0.2 * scale
y1 = 0.15 * scale
z = 0.4 * scale
camera_vertices = [[[-x1, y1, 0],[-x1, -y1, 0],[-x1, -y1, z],[-x1, y1, z],[-x1, y1, 0]],
                  [[x1, y1, 0],[x1, -y1, 0],[x1, -y1, z],[x1, y1, z],[x1, y1, 0]],
                  [[-x1, y1, 0],[-x1, -y1, 0],[x1, -y1, 0],[x1, y1, 0],[-x1, y1, 0]],
                  [[-x1, y1, z],[-x1, -y1, z],[x1, -y1, z],[x1, y1, z],[-x1, y1, z]],
                  [[-x1, -y1, 0],[x1, -y1, 0],[x1, -y1, z],[-x1, -y1, z],[-x1, -y1, 0]]]
r = 0.1* scale

camera_hole_vertices = [[[r*np.cos(teta * np.pi / 180.),y1+0.01,z/2+r*np.sin(teta * np.pi / 180.)] for teta in range(0, 360, 10)]]
T = np.array([0.0, 0.12, 0.0])
camera_hole_vertices = transform_vertices_groups(camera_hole_vertices, T=T)
camera_vertices = transform_vertices_groups(camera_vertices, T=T)

R = np.array([[0, -1, 0],[1, 0, 0],[0, 0, 1]])
T = np.array([0.45, 0.0875, 0.015])
Anemo_vertices1A = transform_vertices_groups(Anemo_vertices,T=T)
Anemo_vertices1B = transform_vertices_groups(Anemo_vertices,R=R,T=T)
T = np.array([-0.45, 0.0875, 0.015])
Anemo_vertices2A = transform_vertices_groups(Anemo_vertices,T=T)
Anemo_vertices2B = transform_vertices_groups(Anemo_vertices,R=R,T=T)

theta = 90
Rp = np.array([[np.cos(theta*np.pi/180), -np.sin(theta*np.pi/180), 0],
               [np.sin(theta*np.pi/180), np.cos(theta*np.pi/180), 0],[0, 0, 1]])

zz = 0.15
Tb = np.array([0.0, 0, zz])
drone_center = transform_vertices_groups(arc2d_vertices(0.1,n=8), T=Tb)
Tp = np.array([0.28, 0, zz])
props = transform_vertices_groups(arc2d_vertices(0.12),T=Tp)
Tp = np.array([-0.28, 0, zz])
props.extend(transform_vertices_groups(arc2d_vertices(0.12), T=Tp))
Tp = np.array([0, -0.28, zz])
props.extend(transform_vertices_groups(arc2d_vertices(0.12), T=Tp))
Tp = np.array([0, 0.28, zz])
props.extend(transform_vertices_groups(arc2d_vertices(0.12), T=Tp))
uav_pole =   [[[-0.28, -0.007, -0.015],[0.28, -0.007, -0.015],[0.28, -0.007, 0.015],[-0.28, -0.007, 0.015]],
            [[-0.28, 0.007, -0.015],[0.28, 0.007, -0.015],[0.28, 0.007, 0.015],[-0.28, 0.007, 0.015]],
            [[-0.28, -0.007, -0.015],[0.28, -0.007, -0.015],[0.28, 0.007, -0.015],[-0.28, 0.007, -0.015]],
            [[-0.28, -0.007, 0.015],[0.28, -0.007, 0.015],[0.28, 0.007, 0.015],[-0.28, 0.007, 0.015]]]
uav_pole = transform_vertices_groups(uav_pole)

uav_pole.extend(transform_vertices_groups(uav_pole,R=Rp))

theta = 45
Rp = np.array([[np.cos(theta*np.pi/180), -np.sin(theta*np.pi/180), 0],
               [np.sin(theta*np.pi/180), np.cos(theta*np.pi/180), 0],[0, 0, 1]])

props=transform_vertices_groups(props, R=Rp)
uav_pole=transform_vertices_groups(uav_pole, R=Rp,T=Tb)

def RoverBox3D(ax,**kwargs):
    kwargs['T'] = kwargs['T']+[0,-0.0*kwargs['scale'],0]
    print(kwargs['T'])  
    roverBox3D(ax,**kwargs)

def roverBox3D(ax,**kwargs):
    poly8 = Poly3DCollection(transform_vertices_groups(props,**kwargs), alpha=0.05, color = "black")
    poly9 = Poly3DCollection(transform_vertices_groups(uav_pole,**kwargs), alpha=0.2, color = "black")
    poly10 = Poly3DCollection(transform_vertices_groups(drone_center,**kwargs), alpha=0.2, color = "black")
    
    poly0 = Poly3DCollection(transform_vertices_groups(Rover_vertices1,**kwargs), alpha=0.5, color = "slategrey")
    poly5 = Poly3DCollection(transform_vertices_groups(camera_hole_vertices,**kwargs), alpha=0.9, color = "black")
    poly6 = Poly3DCollection(transform_vertices_groups(camera_vertices,**kwargs), alpha=0.6, color = "red")

    poly1 = Poly3DCollection(transform_vertices_groups(Anemo_vertices1A,**kwargs), alpha=0.6, color = "red")
    poly2 = Poly3DCollection(transform_vertices_groups(Anemo_vertices1B,**kwargs), alpha=0.6, color = "red")
    
    poly3 = Poly3DCollection(transform_vertices_groups(Anemo_vertices2A,**kwargs), alpha=0.6, color = "blue")
    poly4 = Poly3DCollection(transform_vertices_groups(Anemo_vertices2B,**kwargs), alpha=0.6, color = "blue")

    
    
    #uav
    ax.add_collection3d(poly8)
    ax.add_collection3d(poly9)
    ax.add_collection3d(poly10)

    #Rover Box
    ax.add_collection3d(poly0)
    #Anemo 
    ax.add_collection3d(poly1)
    ax.add_collection3d(poly2)
    ax.add_collection3d(poly3)
    ax.add_collection3d(poly4)
    #camera
    ax.add_collection3d(poly5)
    ax.add_collection3d(poly6)
    
    #W = np.array(transform_vertices_groups([[[Wind[0],Wind[1],Wind[2]]]],**kwargs)[0][0])/S
    PlotWind(ax,**kwargs)
    # ax.quiver(AnemoPos1[0][0][0],AnemoPos1[0][0][1],AnemoPos1[0][0][2], W[0], W[1], W[2],color='blue')
    # ax.quiver(AnemoPos1[1][0][0],AnemoPos1[1][0][1],AnemoPos1[1][0][2], W[0], W[1], W[2],color='red')
    #plotArrow3D(ax, P=AnemoPos1[0][0], Q=W,scale=0.8,color='blue')
    #plotArrow3D(ax, P=AnemoPos1[1][0], Q=W,scale=0.8,color='red')
    # ax.scatter(p[0],p[1],p[2], color="black", s=10)
    #ax.scatter(AnemoPos1[1][0][0],AnemoPos1[1][0][1],AnemoPos1[1][0][2], color="blue", s=10)
    #ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)
    

    
    
    
def PlotWind(ax,**kwargs):
    # Anemometer wind directions
    w1 = np.array([0,0,0])#kwargs['ane1']
    w2 = np.array([0,0,0])#kwargs['ane2']
    try:
        w1=kwargs['ane1']
        w2=kwargs['ane2']
    except:
        pass
    P_blue0 = transform_wind(np.array([0,0,0]), np.array([-0.45, 0.08, 0.06]), **kwargs)
    P_blue1 = transform_wind(w1, np.array([-0.45, 0.08, 0.06]), **kwargs)
    #ax.scatter(P_blue1[0],P_blue1[1],P_blue1[2], color="black", s=10)
    #ax.scatter(P_blue0[0],P_blue0[1],P_blue0[2], color="black", s=10)
    P_red0 = transform_wind(np.array([0,0,0]), np.array([0.45, 0.08, 0.06]), **kwargs)
    P_red1 = transform_wind(w2, np.array([0.45, 0.08, 0.06]), **kwargs)
    plotArrow3D(ax, P=P_blue0, Q=P_blue1,size=7,color='blue',label='Anemometer 1')
    plotArrow3D(ax, P=P_red0, Q=P_red1,size=7,color='red',label='Anemometer 2')


def PlotBowWind(**kwargs):
    # Anemometer wind directions
    w1 = np.array([0,0,0])#kwargs['ane1']
    try:
        w1=kwargs['base_ane']
    except:
        pass
    P_0 = transform_wind(np.array([0,0,0]), np.array([0,0,0]), **kwargs)
    P_1 = transform_wind(w1, np.array([0,0,0]), **kwargs)
    #ax.scatter(P_blue1[0],P_blue1[1],P_blue1[2], color="black", s=10)
    #ax.scatter(P_blue0[0],P_blue0[1],P_blue0[2], color="black", s=10)
    #plotArrow3D(ax, P=P_0, Q=P_1,size=7,color='green',label='Bow Anemometer')
    plt.arrow(x=P_0[0], y=P_0[1], dx=P_1[0], dy=P_1[1], width=.08, facecolor='red', edgecolor='none') 

