import cv2
import numpy as np
import matplotlib.pyplot as plt
from . import pointsOnALine, cat_names

def plot_est_axis(p, lw = 1, ms = 1, ss = 10):
    plt.scatter(p[0,0], p[0,1],marker="o", s = ss, color="orangered")
    plt.plot(p[4:6,0], p[4:6,1], markersize=ms, color="red", linewidth=lw)
    plt.plot(p[2:4,0], p[2:4,1], markersize=ms, color="green", linewidth=lw)
    plt.plot(p[0:2,0], p[0:2,1], markersize=ms,color="blue", linewidth=lw)
    

def plot_gt_axis(p, lw = 1, ms = 1, ss = 10):
    plt.plot(p[4:6,0], p[4:6,1], markersize=ms, color="darkred", linewidth=lw)
    plt.plot(p[2:4,0], p[2:4,1], markersize=ms, color="darkgreen", linewidth=lw)
    plt.plot(p[0:2,0], p[0:2,1], markersize=ms,color="darkblue", linewidth=lw)
    plt.scatter(p[0,0], p[0,1],marker="o", s = ss, color="blue")

def plot_est_axis_oc(p, color): #one color
    plt.plot(p[4:6,0], p[4:6,1], markersize=1, color=color, linewidth=0.7)
    plt.plot(p[2:4,0], p[2:4,1], markersize=1, color=color, linewidth=0.7)
    plt.plot(p[0:2,0], p[0:2,1], markersize=1,color=color, linewidth=0.7)
    plt.scatter(p[0,0], p[0,1],marker="o", s = 8, color=color)

def plot_gt_axis_oc(p, color): #one color
    plt.plot(p[4:6,0], p[4:6,1], markersize=1, color=color, linewidth=0.7)
    plt.plot(p[2:4,0], p[2:4,1], markersize=1, color=color, linewidth=0.7)
    plt.plot(p[0:2,0], p[0:2,1], markersize=1,color=color, linewidth=0.7)
    plt.scatter(p[0,0], p[0,1],marker="o", s = 10, color=color)

def plot_points(p, clr = "Green", size=30):
    plt.scatter(p[:,0], p[:,1],marker="o", s = size, color=clr)

def plot_3dbox(p):
    plt.plot(p[:,0], p[:,1], markersize=1, color="green", linewidth=0.5)

def plot_line(p, clr = "Green"):
    plt.plot(p[:,0], p[:,1], markersize=1,color=clr, linewidth=1)

def plot_fourPoints(p):
    plt.scatter(p[0:1,0], p[0:1,1],marker="o", s = 20, color="Darkred")
    plt.scatter(p[1:2,0], p[1:2,1],marker="o", s = 20, color="Red")
    plt.scatter(p[2:3,0], p[2:3,1],marker="o", s = 20, color="Orange")
    plt.scatter(p[3:4,0], p[3:4,1],marker="o", s = 20, color="Yellow")

def plot_boxLines(p, Color = "mix", label='', linewidth=1, markersize=1):
    color = ["Darkred", "Red", "darksalmon", "Orange", "gold", "greenyellow", "limegreen", "darkgreen", "cyan", "blue", "navy", "magenta" ]
    for i, c in zip(range(12),color):
        if Color == "mix":
            color = c
        else:
            color = Color
        p_L, _, _ = pointsOnALine(p,i)
        
        if i==0:
            plt.plot(p_L[:,0], p_L[:,1], markersize=markersize, color=color, linewidth=linewidth, label=label)
        else:
            plt.plot(p_L[:,0], p_L[:,1], markersize=markersize, color=color, linewidth=linewidth,)

def plot_int_corners(p, Color = "mix", label='', S = 10):
    color = ["Darkred", "Red", "darksalmon", "Orange", "gold", "greenyellow", "limegreen", "darkgreen", "cyan", "blue", "navy", "magenta" ]
    for i, c in zip(range(12),color):
        if Color == "mix":
            color = c
        else:
            color = Color
        _, _, p_I = pointsOnALine(p,i)
        
        if i==0:
            plt.scatter(p_I[:,0], p_I[:,1], marker="x", s = S, color=color, label=label)
        else:
            plt.scatter(p[i,0], p[i,1],marker="o", s = S, color=color)

def plot_corners(p, Color = "mix", label='', S = 10):
    color = ["magenta", "blue", "green", "darkgreen", "limegreen", "gold", "Orange", "Red" ]
    for i, c in zip(range(8),color):
        if Color == "mix":
            color = c
        else:
            color = Color
        if i==0:
            plt.scatter(p[i,0], p[i,1],marker="o", s = S, color=color, label=label)
        else:
            plt.scatter(p[i,0], p[i,1],marker="o", s = S, color=color)
        

def plot_box(p, clr_lines = "mix", clr_corners = "mix", clr_int_corners = "mix", lines = True, corners = True, int_corners = True, one_clr = "none", label = '', linewidth=1, markersize=1):
    if one_clr != "none":
        clr_lines = one_clr
        clr_corners = one_clr
        clr_int_corners = one_clr
    if lines:
        plot_boxLines(p, Color = clr_lines, label=label, markersize=markersize, linewidth=linewidth)
    if corners:
        plot_corners(p, Color = clr_corners, label=label, S=7)
    if int_corners:
        plot_int_corners(p, Color = clr_int_corners, label=label, S=7)

def plot_cat_distribution(cat_names, cat_bools_log, Dir):
    counts=np.sum(cat_bools_log, axis=0)
    
    fig, ax = plt.subplots()
    
    bar_colors = ['green', 'red', 'lightgreen', 'orange', 'cyan', 'blue', 'magenta', 'lightblue', 'tan', 'brown', 'yellow', 'black'] 
    
    rects1 = ax.bar(cat_names[0:len(counts)], counts, color=bar_colors)

    ax.set_ylabel('No. images which cat id has got appered')
    ax.set_title('{:d} synthetic images in the dataset'.format(cat_bools_log.shape[0]))
    ax.bar_label(rects1)
    #fig.tight_layout()
    plt.xticks(rotation=90)
    plt.autoscale()
    plt.savefig(Dir+"cat_dist.png", bbox_inches = "tight")

def plot_cat_key_distribution(cat_names, cat_keys, Dir):
    keypoints = ['0', '1', '2', '3','4', '5', '6', '7', '8']
    for i in range(cat_keys.shape[1]):
        fig, ax = plt.subplots()
        #cat_keys occurrences of each unique value
        (values,counts) = np.unique(cat_keys[:,i], return_counts=True)
        rects1 = ax.bar(values, counts)
        ax.set_ylabel('No. images')
        ax.set_xlabel('No. keypoints in image bound')
        ax.set_title('{} category'.format(cat_names[i]))
        x = np.arange(len(keypoints))
        ax.set_xticks(x, keypoints)
        ax.bar_label(rects1)
        plt.savefig(Dir+"{}_{}_key_dist.png".format(i+1,cat_names[i]))

def plotArea(p, alpha = 0.4, one_clr = "green"):
    P_hull = cv2.convexHull(np.expand_dims(p, axis=1).astype(np.int32), False)[:,0,:]
    plt.fill(P_hull[:,0], P_hull[:,1], alpha = alpha, color = one_clr)
