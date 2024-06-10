import os
import cv2
import numpy as np
from numpy import linalg as L2
import pandas as pd
import pymap3d as pm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import json
import argparse
from configparser import ConfigParser, ExtendedInterpolation

from plotlib.Qplot.bodies import *
from plotlib.ship import *
from plotlib.getONR import calcONRfromData
from tools.misc import *
from plotlib.rover import *
from plotlib.genHws import *
from plotlib.projections import *
from plotlib.Map.plotmap import *
from tools.misc_fucntions import *
from calcHwGPS import *

def plotGPSTraj_rtk(ax, rtk_X, rtk_X_Y):
    #plot3D_WorldArrowAxes(ax,scale=2) 
    ship3D(ax)
    #sea3D(ax)
    ax.scatter(0,0,0.25, color="Red", s=10, label = 'referance', marker='+')
    ax.scatter(rtk_X[0,0],rtk_X[0,1],rtk_X[0,2], color="orange", s=5, label = 'starting point')
    #ax.scatter(rtk_X_Y[0,0],rtk_X_Y[0,1],rtk_X_Y[0,2], color="orange", s=5, label = 'starting point')
    #ax.scatter(onr.rtk_X_cam[0,loc],onr.rtk_X_cam[1,loc],onr.rtk_X_cam[2,loc], color="lime", s=10, label = 'chosen point')
    #ax.plot3D(rtk_X[:,0],rtk_X[:,1],rtk_X[:,2],label='RTK GPS position trajectory', color = 'blue')
    #ax.plot3D(rtk_X_Y[:,0],rtk_X_Y[:,1],rtk_X_Y[:,2],label='RTK GPS position trajectory', color = 'gray')

    ax.plot3D(rtk_X_Y[:,0],rtk_X_Y[:,1],rtk_X_Y[:,2],label='RTK GPS position trajectory', color = 'blue')

    #ax.plot3D(Hw_est_syn_xyz[:,0],Hw_est_syn_xyz[:,1],Hw_est_syn_xyz[:,2], label='TNN Est. position from GPS_to_Syn images',  color = 'magenta')
    # loc = 50
    # u = 950
    # ax.scatter(onr.rtk_X[0,0],onr.rtk_X[1,0],onr.rtk_X[2,0], color="r", s=5, label = 'starting point')
    # ax.scatter(onr.rtk_X_cam[0,loc],onr.rtk_X_cam[1,loc],onr.rtk_X_cam[2,loc], color="lime", s=10, label = 'chosen point')
    # ax.plot3D(onr.rtk_X[0,u:],onr.rtk_X[1,u:],onr.rtk_X[2,u:],label='RTK GPS position', color = 'blue')
        
def plotGPSTraj_rtk_cam(ax, rtk_X_cam):
    #plot3D_WorldArrowAxes(ax,scale=2) 
    ship3D(ax)
    #sea3D(ax)
    ax.scatter(0,0,0.25, color="Red", s=10, label = 'referance', marker='+')
    ax.scatter(rtk_X_cam[0,0],rtk_X_cam[0,1],rtk_X_cam[0,2], color="orange", s=5, label = 'starting point')
    #ax.scatter(rtk_X_Y[0,0],rtk_X_Y[0,1],rtk_X_Y[0,2], color="orange", s=5, label = 'starting point')
    #ax.scatter(onr.rtk_X_cam[0,loc],onr.rtk_X_cam[1,loc],onr.rtk_X_cam[2,loc], color="lime", s=10, label = 'chosen point')
    #ax.plot3D(rtk_X[:,0],rtk_X[:,1],rtk_X[:,2],label='RTK GPS position trajectory', color = 'blue')
    #ax.plot3D(rtk_X_Y[:,0],rtk_X_Y[:,1],rtk_X_Y[:,2],label='RTK GPS position trajectory', color = 'gray')

    ax.plot3D(rtk_X_cam[:,0],rtk_X_cam[:,1],rtk_X_cam[:,2],label='RTK GPS position trajectory', color = 'blue')
   

    #ax.plot3D(Hw_est_syn_xyz[:,0],Hw_est_syn_xyz[:,1],Hw_est_syn_xyz[:,2], label='TNN Est. position from GPS_to_Syn images',  color = 'magenta')
    # loc = 50
    # u = 950
    # ax.scatter(onr.rtk_X[0,0],onr.rtk_X[1,0],onr.rtk_X[2,0], color="r", s=5, label = 'starting point')
    # ax.scatter(onr.rtk_X_cam[0,loc],onr.rtk_X_cam[1,loc],onr.rtk_X_cam[2,loc], color="lime", s=10, label = 'chosen point')
    # ax.plot3D(onr.rtk_X[0,u:],onr.rtk_X[1,u:],onr.rtk_X[2,u:],label='RTK GPS position', color = 'blue')

def plotGPSTraj(ax, Hw):
    #plot3D_WorldArrowAxes(ax,scale=2) 
    ship3D(ax)
    #sea3D(ax)
    ax.scatter(0,0,0.25, color="Red", s=10, label = 'referance', marker='+')
    ax.scatter(Hw[0,3],Hw[1,7],Hw[2,11], color="orange", s=5, label = 'starting point')
    #ax.scatter(onr.rtk_X_cam[0,loc],onr.rtk_X_cam[1,loc],onr.rtk_X_cam[2,loc], color="lime", s=10, label = 'chosen point')
    ax.plot3D(Hw[:,3],Hw[:,7],Hw[:,11],label='RTK GPS position trajectory', color = 'blue')
    #ax.plot3D(Hw_est_syn_xyz[:,0],Hw_est_syn_xyz[:,1],Hw_est_syn_xyz[:,2], label='TNN Est. position from GPS_to_Syn images',  color = 'magenta')
    # loc = 50
    # u = 950
    # ax.scatter(onr.rtk_X[0,0],onr.rtk_X[1,0],onr.rtk_X[2,0], color="r", s=5, label = 'starting point')
    # ax.scatter(onr.rtk_X_cam[0,loc],onr.rtk_X_cam[1,loc],onr.rtk_X_cam[2,loc], color="lime", s=10, label = 'chosen point')
    # ax.plot3D(onr.rtk_X[0,u:],onr.rtk_X[1,u:],onr.rtk_X[2,u:],label='RTK GPS position', color = 'blue')

def plotEsts(ax, cat_names, colors, i):
    if i < len(cat_names):
            ax.plot3D(Hw_ests[:,i,0,3],Hw_ests[:,i,1,3],Hw_ests[:,i,2,3], label= cat_names[i] + ' - TNN position Est. ',  color = colors[i])
    if i == len(cat_names):
        for j in range(len(cat_names)):
            ax.plot3D(Hw_ests[:,j,0,3],Hw_ests[:,j,1,3],Hw_ests[:,j,2,3], label=cat_names[j] + ' - TNN position Est.',  color = colors[j])
        
    if i == len(cat_names)+1:
        for j in range(len(cat_names)):
            ax.plot3D(Hw_ests[:,j,0,3],Hw_ests[:,j,1,3],Hw_ests[:,j,2,3], label=cat_names[j] + ' - TNN position Est.',  color = colors[j])
        #ax.plot3D(Hw_whabas[:,0,0,3],Hw_whabas[:,0,1,3],Hw_whabas[:,0,2,3], label='TNN-whabas position Est.',  color = 'gray')
        ax.plot3D(Hw_bayesian_fusion[:,0,0,3],Hw_bayesian_fusion[:,0,1,3],Hw_bayesian_fusion[:,0,2,3], label='TNN-bayesian_fusion position Est.',  color = 'k')
    if i == len(cat_names)+2:
        #ax.plot3D(Hw_whabas[:,0,0,3],Hw_whabas[:,0,1,3],Hw_whabas[:,0,2,3], label='TNN-whabas position Est.',  color = 'gray') 
        ax.plot3D(Hw_bayesian_fusion[:,0,0,3],Hw_bayesian_fusion[:,0,1,3],Hw_bayesian_fusion[:,0,2,3], label='TNN-bayesian_fusion position Est.',  color = 'k')

def plotmulti(ax, elevation="side", l = np.array([[-20, 20],[-24,10],[-6,8]]), div = 5, zdiv=1):
    
    ax.set_xlim(l[0,0],l[0,1])

    ax.set_ylim(l[1,0],l[1,1])
    ax.set_zlim(l[2,0],l[2,1])
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, 5))
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(start, end, 5))
    start, end = ax.get_zlim()
    ax.zaxis.set_ticks(np.arange(start, end, 5))
    
   
    # ax.set_xticks(np.arange(((l[0,0])//div + 1) * div,l[0,1]//div * div +1, div))
    # ax.set_yticks(np.arange(((l[1,0])//div + 1) * div,l[1,1]//div * div +1, div))
    
    
    # #ax.set_xticks(np.arange(-20, 20, 5))
    # print(np.arange(0,l[0,1]//div * div +1, div))
    # print(np.arange(((l[0,0])//div + 1) * div,0, div))
    
    ax.view_init(azim=0, elev=0)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    if elevation == "side":
        ax.set_ylabel('$y$ (m)')
        ax.set_zlabel('$z$ (m)')
        ax.view_init(azim=0, elev=0)
        ax.set_xticks([])
        ax.set_yticks(np.arange(((l[1,0])//div + 1) * div,l[1,1]//div * div +1, div))
        ax.set_zticks(np.arange(((l[2,0])//zdiv + 1) * zdiv,l[2,1]//zdiv * zdiv +1, zdiv))
        ax.legend(loc='upper right') 
        #ax.set_title('Side View')

    if elevation == "top":
        ax.set_xlabel('$x$ (m)')
        ax.set_ylabel('$y$ (m)')
        ax.set_xticks(np.arange(((l[0,0])//div + 1) * div,l[0,1]//div * div +1, div))
        ax.set_yticks(np.arange(((l[1,0])//div + 1) * div,l[1,1]//div * div +1, div))
        ax.set_zticks([])
        ax.zaxis.line.set_visible(False)
        ax.view_init(azim=-90, elev=90)
        #ax.set_title('Top View')
        
    if elevation == "3d":
        ax.set_ylabel('$y$ (m)')
        ax.set_xlabel('$x$ (m)')
        ax.set_zlabel('$z$ (m)')
        ax.view_init(azim=-30, elev=20)
        ax.set_xticks(np.arange(((l[0,0])//div + 1) * div,l[0,1]//div * div +1, div))
        ax.set_yticks(np.arange(((l[1,0])//div + 1) * div,l[1,1]//div * div +1, div))
        ax.set_zticks(np.arange(((l[2,0])//zdiv + 1) * zdiv,l[2,1]//zdiv * zdiv +1, zdiv))

def error_plots(ax, y1, y2, y3, c = 'k', ls='-',xlabel='Frame No.',ylabel='$x$ (m)', name_shift= 0, label = "TNN"):
    x = np.arange(0, y1.shape[0], 1, dtype=int)+name_shift+1
    ax[0].plot(x, y1, c=c, ls=ls, label=label)
    ax[0].legend(loc="upper right")
    ax[0].set_xticks(x)
    ax[0].tick_params(axis='x',rotation=90, labelsize=2)
    ax[0].grid(True)

    ax[1].plot(x, y2, c=c, ls=ls)
    ax[1].set( ylabel=ylabel)
    ax[1].set_xticks(x)
    ax[1].tick_params(axis='x',rotation=90, labelsize=2)
    ax[1].grid(True)

    ax[2].plot(x, y3, c=c, ls=ls)
    ax[2].set_xticks(x)
    ax[2].tick_params(axis='x',rotation=90, labelsize=2)
    ax[2].grid(True)

    ax[2].set(xlabel=xlabel)
    plt.subplots_adjust(wspace=0.0,hspace=0.4)

# def traj_plots(ax, y, c = 'k', ls='-',xlabel='Frame No.',ylabel='$x$ (m)', name_shift= 0, label = "TNN"):
#     x = np.arange(0, y.shape[0], 1, dtype=int)+name_shift+1
#     ax[0].plot(x, y[:,0,3], c=c, ls=ls, label=label)
#     ax[0].legend(loc="upper right")
#     ax[0].set_xticks(x)
#     ax[0].tick_params(axis='x',rotation=90, labelsize=2)
#     ax[0].grid(True)

#     ax[1].plot(x, y[:,1,3], c=c, ls=ls)
#     ax[1].set( ylabel=ylabel)
#     ax[1].set_xticks(x)
#     ax[1].tick_params(axis='x',rotation=90, labelsize=2)
#     ax[1].grid(True)

#     ax[2].plot(x, y[:,2,3], c=c, ls=ls)
#     ax[2].set_xticks(x)
#     ax[2].tick_params(axis='x',rotation=90, labelsize=2)
#     ax[2].grid(True)

#     ax[2].set(xlabel=xlabel)
#     plt.subplots_adjust(wspace=0.0,hspace=0.4)
    

def error_plots(ax, y1, y2, y3, c = 'k', ls='-',xlabel='Frame No.',ylabel='$x$ (m)', name_shift= 0, label = "TNN"):
    x = np.arange(0, y1.shape[0], 1, dtype=int)+name_shift+1
    ax[0].plot(x, y1, c=c, ls=ls, label=label)
    ax[0].legend(loc="upper right")
    ax[0].set_xticks(x)
    ax[0].tick_params(axis='x',rotation=90, labelsize=2)
    ax[0].grid(True)

    ax[1].plot(x, y2, c=c, ls=ls)
    ax[1].set( ylabel=ylabel)
    ax[1].set_xticks(x)
    ax[1].tick_params(axis='x',rotation=90, labelsize=2)
    ax[1].grid(True)

    ax[2].plot(x, y3, c=c, ls=ls)
    ax[2].set_xticks(x)
    ax[2].tick_params(axis='x',rotation=90, labelsize=2)
    ax[2].grid(True)

    ax[2].set(xlabel=xlabel)
    plt.subplots_adjust(wspace=0.0,hspace=0.4)


def d_plot(ax, D, ls='-',xlabel='Frame No.',ylabel='$d$', name_shift= 0):
    fig, ax = plt.subplots(figsize = (25,12))
    x = np.arange(0, D.shape[0], 1, dtype=int)+name_shift+1
    ax.plot(x, D[:,0], c="red", ls=ls, label="$d_1$")
    ax.plot(x, D[:,1], c="green", ls=ls, label="$d_2$")
    ax.plot(x, D[:,2], c="blue", ls=ls, label="$d_3$")
    #ax.fill_between(x, y[:,i,3] - 2*std_dev[:,i], y[:,i,3] + 2*std_dev[:,i], color=c, alpha=0.2)
    ax.legend(loc="upper right")
    ax.set_xticks(x)
    ax.tick_params(axis='x',rotation=90, labelsize=2)
    ax.grid(True)

    ax.set(ylabel=ylabel)
    ax.set(xlabel=xlabel)
    ax.set_ylim([0.9995, 1.0001])


def traj_plots(ax, y, std_dev, c = 'k', ls='-',xlabel='Frame No.',ylabel='$x$ (m)', name_shift= 0, label = "TNN"):
    std_dev = std_dev.cpu().numpy()
    x = np.arange(0, y.shape[0], 1, dtype=int)+name_shift+1
    for i in range(3):
        ax[i].plot(x, y[:,i,3], c=c, ls=ls, label=label)
        ax[i].fill_between(x, y[:,i,3] - 2*std_dev[:,i], y[:,i,3] + 2*std_dev[:,i], color=c, alpha=0.2)
        ax[i].legend(loc="upper right")
        ax[i].set_xticks(x)
        ax[i].tick_params(axis='x',rotation=90, labelsize=2)
        ax[i].grid(True)

    ax[1].set(ylabel=ylabel)
    ax[2].set(xlabel=xlabel)
    plt.subplots_adjust(wspace=0.0,hspace=0.4)

def error_traj_plots(ax, gps, y, std_dev, c = 'k', ls='-',xlabel='Frame No.',ylabel='$x$ (m)', name_shift= 0, label = "TNN"):
    std_dev = std_dev.cpu().numpy()
    x = np.arange(0, y.shape[0], 1, dtype=int)+name_shift+1
    for i in range(3):
        ax[i].plot(x, gps[:,i]-y[:,i,3], c=c, ls=ls, label=label)
        ax[i].fill_between(x, gps[:,i]-y[:,i,3] - 2*std_dev[:,i], gps[:,i]-y[:,i,3] + 2*std_dev[:,i], color=c, alpha=0.2)
        ax[i].legend(loc="upper right")
        ax[i].set_xticks(x)
        ax[i].tick_params(axis='x',rotation=90, labelsize=2)
        ax[i].grid(True)

    ax[1].set(ylabel=ylabel)
    ax[2].set(xlabel=xlabel)
    plt.subplots_adjust(wspace=0.0,hspace=0.4)

def mae_3d(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Plot trajectories', add_help=False)
    parser.add_argument('--configfile', default='', type=str,
                            help="config file location")
    args = parser.parse_args()

    config = ConfigParser(interpolation=ExtendedInterpolation())
    configtrain = config.read(args.configfile)
    results = config.get('Test', 'results')+"results/"
    testname = config.get('Test', 'testname')
    example_database = config.get('Dir', 'EXAMPLES')
    cat_names = json.loads(config.get('Test', 'Obj_list'))
    colors = json.loads(config.get('Test', 'colors'))

    configtestdir = config.read('test_config.ini')
    examples_names = json.loads(config.get('Test', 'Test_samples'))
    # Test_no = 44

    # TestFolder = "D_Tests_2023.05.05/"


    # DirSub = "Data_collected/"+TestFolder+TestFolder[0]+"_Test_{}/".format(Test_no)
    # DirSub = "/home/maneesh/Desktop/LAB2.0/DATA-FDCL/RawRealCollectedData/01_RR_Test_44/GPS_Test_44/"
    # plots = "/home/maneesh/Desktop/LAB2.0/DATA-FDCL/PrcessedData/" + "plots_Test_{}/".format(Test_no)

    for i, example in enumerate(examples_names):
        DirGPS = example_database + example + "/" + example + "_GPS/"
        plots = results + example + "_" + testname + "/trjectoryplots/" + example +  "_plot/"
        try:
            os.makedirs(plots)
        except:
            pass

        try:
            Hw_Row = np.loadtxt(DirGPS+"Hw_RTKGPS_"+example+".txt")
            rtk_X = np.loadtxt(DirGPS+"rtk_X_"+example+".txt")
            rtk_X_Y = np.loadtxt(DirGPS+"rtk_X_Y_"+example+".txt")
            rtk_X_cam = np.loadtxt(DirGPS+"rtk_X_cam_"+example+".txt")
            time = np.loadtxt(DirGPS+"time_"+example+".txt")
            print('\033[102m' + "GPS Trajectory Loading Completed!" + '\033[0m')

        except:
            print('\033[43m' + " Generating GPS Trajectory" + '\033[0m')
            onr = calcONRfromData(DirGPS)
            
            llcrop_pix1, llzoom1, zbbox1, map1 = plotMap(onr.base_llh, zoom_factor = 90)
            llcrop_pix2, llzoom2, zbbox2, map2 = plotMap(onr.base_llh, zoom_factor = 1.5)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15,9))
            ax1.plot(onr.base_llh[1, :], onr.base_llh[0, :], c='b',linewidth = '1',label='ship trajectory')
            ax1.plot(zbbox2[:,0], zbbox2[:,1], c='g',ls = '--',linewidth = '1', label='zoom')
            ax1.set_title('Plotting llh (lat,lon,hig) Data ')
            ax1.set_xlabel('latitude ($deg$)')
            ax1.set_ylabel('longitude$ ($deg$)')
            ax1.legend()
            #ax1.set_xlim(mapbbox[mapNo-1,0],mapbbox[mapNo-1,1])
            #ax1.set_ylim(mapbbox[mapNo-1,2],mapbbox[mapNo-1,3])
            ax1.imshow(map1, zorder=0, extent = llzoom1, aspect= 'equal')
            ax1.grid()
            
            ax2.imshow(map2, zorder=0, extent = llzoom2, aspect= 'equal')
            ax2.plot(onr.base_llh[1, :], onr.base_llh[0, :], c='b', label='ship trajectory')
            ax2.scatter(onr.base_llh[1, 0], onr.base_llh[0, 0], label='start', c='g')
            ax2.scatter(onr.base_llh[1, -1], onr.base_llh[0, -1], label='end', c='r')
            ax2.set_xlabel('latitude ($deg$)')
            ax2.set_ylabel('longitude ($deg$)')
            #ax2.axis('equal')
            ax2.legend()
            ax2.set_title('Plotting llh Data ')
            ax2.grid()
            
            plt.savefig(plots+example+"_llh.png")
            try:
                Hw_Row, Hi = calHwfromGPS_cam(onr)
                rtk_X_cam = onr.rtk_X_camV2.T
                np.savetxt(DirGPS+"rtk_X_cam_"+ example +".txt",rtk_X_cam)
            except:
                Hw_Row, Hi = calHwfromGPS(onr)
           
            rtk_X = onr.rtk_X.T
            rtk_X_Y = onr.rtk_X_Y.T
            time = onr.time
            Time = onr.Time
            
            #np.savetxt("/home/maneesh/Desktop/Syn_Ship_Data_1/Tracker/Hw.txt",Hw_Row)
            np.savetxt(DirGPS+"Hw_RTKGPS_"+ example +".txt",Hw_Row)
            np.savetxt(DirGPS+"rtk_X_"+ example +".txt",rtk_X)
            np.savetxt(DirGPS+"rtk_X_Y_"+ example +".txt",rtk_X_Y)
            np.savetxt(DirGPS+"time_"+ example +".txt",time)
            np.savetxt(DirGPS+"Time_"+ example +".txt",Time)
            # 
            print('\033[102m' + "GPS Trajectory Generated!" + '\033[0m')



        cat_ID = 1
        img_no = 145
        mapID = np.array([5, 8, 6, 10, 1, 4, 3, 9, 0, 2, 8])
        print(results + example + "_" + testname + '/pts/Hw_'+ example +'_'+testname+'.pt')
        print("/home/maneesh/Desktop/LAB2.0/DATA-FDCL/CheckpointsE01/E01rev0_12-25-2023_Multi-6X-H332k.NH102k__DETR-1.1-NQ6_NC6__DT435454/results/E_Test_5_E01rev0_12-25-2023_Multi-6X-H332k.NH102k__DETR-1.1-NQ6_NC6__DT435454/pts")
        Tests = torch.load(results + example + "_" + testname + '/pts/Hw_'+ example +'_'+testname+'.pt')
        

        # "Weights_inliers": Weights_inliers, "Hws_inliers": Hws_inliers, "Hws_inliers_BF": Hws_inliers_BF,
		# "Weights_all": Weights_all, "Hws_all": Hws_all, "Hws_all_BF": Hws_all_BF},

        # Weights = Tests["Weights_all"]
        Hw_ests = Tests["Hws_all"]
        Hw_bayesian_fusion = Tests["Hws_inliers_BF"]
        std_dev = Tests["std_dev_P_inliers"]
        D_inliers = Tests["D_inliers"]
        
       
        # print(Hw_ests.size())
        # print(Hw_ests[0,0,:,:])
        # print(Hw_ests[0,0,0,3])
        # print(Hw_ests[0,0,1,3])
        # print(Hw_ests[0,0,2,3])
        # print(Hw_whabas.size())
        # print(Hw_bayesian_fusion.size())

        #'''
        for i in range(len(cat_names)+3):
            fig = plt.figure(figsize = (20,12),constrained_layout=True)  
            #fig.tight_layout()
            #ax = fig.subplots_adjust(left=-0.11,right=-0.5)
            ax = fig.add_subplot(121, projection='3d')
            #plotGPSTraj(ax, Hw_Row)
            try:
                plotGPSTraj_rtk(ax, rtk_X, rtk_X_Y)
            except:
                pass

            try:
                plotEsts(ax, cat_names, colors, i)
            except:
                pass

            plotmulti(ax, elevation="top",l = np.array([[-20, 20],[-24,10],[-6,8]]),div = 5, zdiv=2)


            ax = fig.add_subplot(122, projection='3d')
            #plotGPSTraj(ax, Hw_Row)
            # plotGPSTraj_rtk(ax, rtk_X, rtk_X_Y, rtk_X_cam)
            # plotEsts(ax, cat_names, colors, i)
            try:
                plotGPSTraj_rtk(ax, rtk_X, rtk_X_Y)
            except:
                pass
            try:
                plotEsts(ax, cat_names, colors, i)
            except:
                pass
            plotmulti(ax, elevation="side", l = np.array([[-15, 15],[-15,3],[-6,8]]),div = 2, zdiv=2)
            ax.set_facecolor('white')

            # set the spacing between subplots
            plt.subplots_adjust(left=0.1,
                                bottom=0.1,
                                right=0.9,
                                top=0.9,
                                wspace=0.0,
                                hspace=0.0)
            if i < len(cat_names):
                fig.suptitle('Camera Relative Pose estimation for '+ cat_names[i] + ' object vs RTK-GPS+IMU' )
                plt.savefig(plots+"/" + str(i+1)+ "_" +cat_names[i]+ "__" + testname +".png", dpi=300)
                #plt.savefig(plots+"/s"+ str(i+1)+ "_" +cat_names[i]+ "__" +testname +".svg", dpi=300)

            if i == len(cat_names):
                fig.suptitle('Camera Relative Pose estimations from multiple objects vs RTK-GPS+IMU' )
                plt.savefig(plots+"/" + str(i+1)+ "_All__" + testname +".png", dpi=300)
                #plt.savefig(plots+"/s"+ str(i+1)+ "_All__" +testname +".svg", dpi=300)

            # if i == len(cat_names):
            #     fig.suptitle('Camera Relative Pose estimations from multiple objects + fusion vs RTK-GPS+IMU' )
            #     plt.savefig(plots+"/" + str(i+1)+ "_All+W_" + testname +".png", dpi=300)
            #     plt.savefig(plots+"/s"+ str(i+1)+ "_All+W_" +testname +".svg", dpi=300)

            if i == len(cat_names)+1:
                fig.suptitle('Camera Relative BF Pose estimation vs RTK-GPS+IMU')
                plt.savefig(plots+"/" + str(i+1)+ "_TNN-fusion__" + testname +".png", dpi=300)
                #plt.savefig(plots+"/s"+ str(i+1)+ "_TNN-fusion__" +testname +".svg", dpi=300)
        #'''
        
        try:
            print("Saving error plots")
            d_plot(ax, D_inliers, ls='-',xlabel='Frame No.',ylabel='$d$', name_shift= 0)
            plt.savefig(plots+"/D_" + testname +".png",bbox_inches='tight', dpi=300)
            fig, ax = plt.subplots(figsize = (25,12),nrows=3, ncols=1)
            #Hw_ests_whabas = Hw_whabas.squeeze(1).numpy()
            Hw_ests_bayesian_fusion = Hw_bayesian_fusion.squeeze(1).numpy()
            frame_shift = 0
            frame_del = -18
            #Hw_ests_corrected_whabas = Hw_ests_whabas#[frame_shift:frame_del,:]
            Hw_ests_corrected_bayesian_fusion = Hw_ests_bayesian_fusion
            print("GPS points : ", rtk_X_cam.shape)
            #print("camera points whabas: ", Hw_ests_corrected_whabas.shape)
            print("camera points bayesian fusion: ", Hw_ests_corrected_bayesian_fusion.shape)
            
            #error_plots(ax, Hw_ests_corrected_whabas[:,0,3], Hw_ests_corrected_whabas[:,1,3], Hw_ests_corrected_whabas[:,2,3], c = 'red', ls='-',name_shift=frame_shift, label="TNN-est")
            #traj_plots(ax, Hw_ests_corrected_bayesian_fusion, c = 'blue', ls='-',name_shift=frame_shift, label="TNN-est")
            traj_plots(ax, Hw_ests_corrected_bayesian_fusion, std_dev, c = 'blue', ls='-',name_shift=frame_shift, label="TNN-est")
            #error_plots(ax, Hw_ests_corrected[:,0,3], Hw_ests_corrected[:,1,3], Hw_ests_corrected[:,2,3], c = 'blue', ls='-')
            error_plots(ax, rtk_X_cam[:,0], rtk_X_cam[:,1], rtk_X_cam[:,2], c = 'k', ls='--',name_shift=frame_shift, label="RTK-GPS")
            
            plt.savefig(plots+"/x_GPS-TNN-BF__" + testname +".png",bbox_inches='tight', dpi=300)
            #plt.savefig(plots+"/x_GPS-TNN-whabas__" + testname +".svg",bbox_inches='tight', dpi=300)

            fig, ax = plt.subplots(figsize = (25,12),nrows=3, ncols=1)
            #error_plots(ax, rtk_X_cam[:,0]-Hw_ests_corrected_whabas[:,0,3], rtk_X_cam[:,1]-Hw_ests_corrected_whabas[:,1,3], rtk_X_cam[:,2]-Hw_ests_corrected_whabas[:,2,3],ylabel='$e_x$ (m)', c = 'red', ls='-', label="$x_{RTK-GPS}-x_{TNN-est}$")
            #error_plots(ax, rtk_X_cam[:,0]-Hw_ests_corrected_bayesian_fusion[:,0,3], rtk_X_cam[:,1]-Hw_ests_corrected_bayesian_fusion[:,1,3], rtk_X_cam[:,2]-Hw_ests_corrected_bayesian_fusion[:,2,3],ylabel='$e_x$ (m)', c = 'blue', ls='-', label="$x_{RTK-GPS}-x_{TNN-est}$")
            error_traj_plots(ax, rtk_X_cam, Hw_ests_corrected_bayesian_fusion, std_dev, ylabel='$e_x$ (m)', c = 'blue', ls='-', label="$x_{RTK-GPS}-x_{TNN-est}$")
           
            plt.savefig(plots+"/ex_GPS-TNN-BF__" + testname +".png",bbox_inches='tight', dpi=300)
            #plt.savefig(plots+"/ex_GPS-TNN-whabas__" + testname +".svg",bbox_inches='tight', dpi=300)

            np.savetxt(results + example + "_" + testname + '/pts/x_RTK_GPS_'+ example +'_'+testname+'.txt', rtk_X_cam)
            #np.savetxt(results + example + "_" + testname + '/pts/x_TNN_whabas_'+ example +'_'+testname+'.txt', Hw_ests_corrected_whabas[:,:,3])
            np.savetxt(results + example + "_" + testname + '/pts/x_TNN_bayesian_fusion_'+ example +'_'+testname+'.txt', Hw_ests_corrected_bayesian_fusion[:,:,3])

            print("finished!")
        except:
            pass

