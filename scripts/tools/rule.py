import cv2
import numpy as np
from . import projectedKnown3dBoxPoints, cat_points, knownWorld3dBoxPoints

def is_points_image_bound(Pc):
    if (np.any(Pc[:,0]<=0) or np.any(Pc[:,0]>=640) or np.any(Pc[:,1]<=0) or np.any(Pc[:,1]>=480)):
        return False
    else:
        return True

def is_object_size_good(Pc,Thr_x, Thr_y):
    if np.max(Pc[:,0])-np.min(Pc[:,0]) < Thr_x or np.max(Pc[:,1])-np.min(Pc[:,1]) < Thr_y :
        return False
    else:
        return True

def atleast_more_objectpoints_visible(Pc):
    count = 0
    for i in range(8):
        if Pc[i,0] >= 0 and Pc[i,0] <= 640 and Pc[i,1] >= 0 and Pc[i,1] <= 480:
            count+=1
    if count >= 6:
        return count, True
    else:
        return count, False

def is_object_visible(Pc, Thr_x=20, Thr_y=12):
    if is_points_image_bound(Pc):
        count = 8
        if is_object_size_good(Pc, Thr_x, Thr_y):
            return count, True
        else:
            return count, False
    else:
        count, Bool =  atleast_more_objectpoints_visible(Pc)
        #print(count)
        return count, Bool

def categorize_objects(max_cat_id, Hc, K, Thr_x, Thr_y, points_only):
    cat_ids = []
    Pcs = []
    Pws = []
    img_cat_bool_log = np.zeros(max_cat_id) 
    img_cat_keys_log = np.zeros(max_cat_id) 
    for cat_id_i in range(1, max_cat_id+1):
        Pw = cat_points(cat_id=cat_id_i)
        Pc, _, _, _, _ = projectedKnown3dBoxPoints(Pw, Hc, K, points_only=points_only)
        Pw, _, _, _, _ = knownWorld3dBoxPoints(Pw, points_only = points_only)
        count, Bool = is_object_visible(Pc, Thr_x, Thr_y)
        img_cat_keys_log[cat_id_i-1] = count
        if Bool:
            cat_ids.append(cat_id_i)
            Pcs.append(Pc)
            Pws.append(Pw)
            img_cat_bool_log[cat_id_i-1] = 1
            
    img_cat_bool_log = np.expand_dims(img_cat_bool_log, axis=0)
    img_cat_keys_log = np.expand_dims(img_cat_keys_log, axis=0)
    
    return cat_ids, Pcs, Pws, img_cat_bool_log, img_cat_keys_log

def categorize_objects_2(max_cat_id, Hc, K, Thr_x, Thr_y, points_only):
    cat_ids = []
    Pcs = []
    Pws = []
    Areas = []
    Radius = []
    img_cat_bool_log = np.zeros(max_cat_id) 
    img_cat_keys_log = np.zeros(max_cat_id) 
    for cat_id_i in range(1, max_cat_id+1):
        Pw = cat_points(cat_id=cat_id_i)
        Pc, _, _, _, _ = projectedKnown3dBoxPoints(Pw, Hc, K, points_only=points_only)
        Pw, _, _, _, _ = knownWorld3dBoxPoints(Pw, points_only = points_only)
        count, Bool = is_object_visible(Pc, Thr_x, Thr_y)
        img_cat_keys_log[cat_id_i-1] = count
        if Bool:
            cat_ids.append(cat_id_i)
            Pcs.append(Pc)
            Pws.append(Pw)
            P_hull = cv2.convexHull(np.expand_dims(Pc, axis=1).astype(np.int32), False)[:,0,:]
            Area = cv2.contourArea(P_hull)
            (x_axis,y_axis), radius = cv2.minEnclosingCircle(P_hull)
            Areas.append(Area)
            Radius.append(radius)

            img_cat_bool_log[cat_id_i-1] = 1
    
    img_cat_bool_log = np.expand_dims(img_cat_bool_log, axis=0)
    img_cat_keys_log = np.expand_dims(img_cat_keys_log, axis=0)

    try:
        Areas = np.array(Areas)        
        arrmax, arrmin = Areas.max(), Areas.min()
        Areas = 1 + (Areas-arrmin) / (arrmax-arrmin)

        Radius = np.array(Radius)
        rrmax, rrmin = Radius.max(), Radius.min()
        Radius = 1 + (Radius-rrmin) / (rrmax-rrmin)
    except:
        Areas = np.array([1.0])
        Radius = np.array([1.0])
    
    return cat_ids, Pcs, Pws, Areas, Radius, img_cat_bool_log, img_cat_keys_log