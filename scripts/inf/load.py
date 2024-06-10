import numpy as np
import torch
from .Epnp import *
from .projections import *
#from .wahbas import *
from .BF import bayesian_fusion
from .pnp import *


def model_pred_naive(im, model, transform, num_keys, confidance = 0.95):
	# mean-std normalize the input image (batch-size: 1)
	img = transform(im).unsqueeze(0)
	outputs = model(img) #use trained model for estimate outputs
     
	# Use softmax for predictions logits to get confidance values
	predictions = outputs['pred_logits'].softmax(-1)[0, :, :-1]
	confidance_val = predictions.max(-1).values
	keep = confidance_val > confidance  #keep is tensor([ True, False, False,  True, ... ]) which is used to filter output based on confidance

	keypoints_keep = outputs['pred_keypoints'][0, keep] # outputs['pred_keypoints'].size() = [1, NQ, 64]
	keypoints_all = outputs['pred_keypoints'][0]

	catIDs_keep = torch.from_numpy(np.expand_dims(np.array([p.argmax().numpy()+1 for p in predictions[keep]]), axis=1))
	catIDs_all = torch.from_numpy(np.expand_dims(np.array([p.argmax().numpy()+1 for p in predictions]), axis=1))

	w, h = im.size
	scaleup_keep = torch.tensor([w, h] * num_keys, dtype = torch.float32).repeat(keypoints_keep[:,:num_keys*2].shape[0],1)
	scaleup_all = torch.tensor([w, h] * num_keys, dtype = torch.float32).repeat(keypoints_all[:,:num_keys*2].shape[0],1)

	keypoints_keep_scaledup = keypoints_keep[:,:num_keys*2]  *  scaleup_keep
	keypoints_all_scaledup = keypoints_all[:,:num_keys*2]  *  scaleup_all

	Keypoints_keep = keypoints_keep_scaledup.view(-1, num_keys, 2)
	Keypoints_all = keypoints_all_scaledup.view(-1, num_keys, 2)

	outputs = {"Keypoints_all": Keypoints_all,
			   "Keypoints_keep": Keypoints_keep,
			   "catIDs_all": catIDs_all,
			   "catIDs_keep":catIDs_keep,
			   "keep": keep,
			   "confidance_values":confidance_val}
	return outputs

def getIdx(catIDs_filtered, i):
    try:
        return torch.nonzero(catIDs_filtered==i+1)[0], catIDs_filtered
    except:
        for j in range(catIDs_filtered.size()[0]):
            if catIDs_filtered[j] == -1:
                catIDs_filtered[j] = i+1
                break 
        try: 
            return torch.nonzero(catIDs_filtered==i+1)[0], catIDs_filtered 
        except:
            W = '\033[43m' + " Warning >>" + '\033[0m'
            print(W + '\033[100m' + ": object repetition include in predictions even though confidance is high" + '\033[0m')
            catIDs_filtered[i] = i+1
            return torch.nonzero(catIDs_filtered==i+1)[0], catIDs_filtered 
    
def getFilteredNamsIdx(catIDs, keep):
    catIDs_filtered = torch.tensor([catIDs[i] if keep[i] else -1 for i in range(catIDs.size()[0])])
    rearranged_idx = []
    for i in range(catIDs.size()[0]):
        idx, catIDs_filtered = getIdx(catIDs_filtered, i)
        rearranged_idx.append(idx)
    r_idx = torch.stack(rearranged_idx, dim=1)[0]
    return r_idx, catIDs_filtered

def model_pred(im, model, transform, num_keys, confidance = 0.8):
	out = model_pred_naive(im, model, transform, num_keys, confidance = confidance)
	r_idx, catIDs_filtered = getFilteredNamsIdx(out["catIDs_all"].T[0], out["keep"])
	rearranged_keypoints= torch.index_select(out["Keypoints_all"], 0, r_idx)
	rearranged_keep = torch.index_select(out["keep"], 0, r_idx)
	rearranged_confidance_values= torch.index_select(out["confidance_values"], 0, r_idx)
     
	outputs = {"idx": r_idx,
			   "catIDs": catIDs_filtered,
			   "keypoints": rearranged_keypoints,
			   "keep":out["keep"],
			   "rkeep":rearranged_keep,
			   "confidance_values":rearranged_confidance_values}
	return outputs

def detect(im, model, transform, num_keys, Pws, K, cat_IDs_order = np.array([1,0,4,3,7,9,5,2,6,8,10])):
	
	# mean-std normalize the input image (batch-size: 1)
	img = transform(im).unsqueeze(0)

	# demo model only support by default images with aspect ratio between 0.5 and 2
	# if you want to use images with an aspect ratio outside this range
	# rescale your image so that the maximum size is at most 1333 for best results
	assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

	outputs = model(img)
	
	predictions = outputs['pred_logits'].softmax(-1)[0, :, :-1]

	Confidance_val = predictions.max(-1).values

	keep = Confidance_val > 0.95
	keypoints = outputs['pred_keypoints'][0, keep]
	keypoints_raw = outputs['pred_keypoints'][0]

	nameIDs = np.expand_dims(np.array([p.argmax().numpy()+1 for p in predictions[keep]]), axis=1)
	nameIDs_raw = np.expand_dims(np.array([p.argmax().numpy()+1 for p in predictions]), axis=1)

	cat_names = ['dog house','no object class','dog house center','dog house right','whole ship','landing pad','house','dog house left','house long',
	             'super structure','ship stern']

    #keypoints[:, 2:36]
	#C_pred = keypoints[:, :2] # shape (N, 2)
	Z_pred = keypoints[:,:num_keys*2] # shape (N, 64)
	Z_pred_raw = keypoints_raw[:,:num_keys*2]

	w, h = im.size
	scaleup = torch.tensor([w, h] * num_keys, dtype = torch.float32).repeat(Z_pred.shape[0],1)
	scaleup_raw = torch.tensor([w, h] * num_keys, dtype = torch.float32).repeat(Z_pred_raw.shape[0],1)
	#print(Z_pred.size())
	#print(scaleup.size())
	A_pred =  Z_pred  *  scaleup
	A_pred_raw =  Z_pred_raw  *  scaleup_raw
	#print(A_pred.size())
	keypoints_scaled = A_pred.view(-1, num_keys, 2)
	keypoints_scaled_raw = A_pred_raw.view(-1, num_keys, 2)

	Pc3ds = torch.cat((keypoints_scaled_raw, 
           torch.ones(keypoints_scaled_raw.size()[0], 32, 1).to(keypoints_scaled_raw.device)), dim=-1)


	H_pnp = Epnp(keypoints_scaled_raw,Pws,K, dist = None, tensor=True, device= 'cpu', dtype=torch.float32)
	"""correction """
	tt = -180
	tt = tt * np.pi / 180
	R0 = torch.tensor(([[1,0,0],[0,np.cos(tt),-np.sin(tt)],[0,np.sin(tt),np.cos(tt)]]),device=keypoints_scaled_raw.device, dtype=torch.float32).unsqueeze(0)

	Hc_est = torch.matmul(R0,H_pnp)
	# Hcc = batchPnP_Hc(Pws, keypoints_scaled_raw,  K)
	# print("Hcc : ",Hcc)


	Hw_est = Hc2Hw_tensor(Hc_est)
	

	r = 2 #remove null object from the Hw est set
	Hw_est_wno = torch.cat((Hw_est[0:r-1,:,:],Hw_est[r:,:,:]),axis=0) # without null object(wno) / remove null object from the Hw est set
	W = torch.cat((Confidance_val[0:r-1],Confidance_val[r:]),axis=0) #remove null object from the W est set
	Hw_est_wahbas = wahbasSimTransRot(Hw_est_wno.numpy(), W.numpy(), n = 200)
	Hw_est_wahbas = np.expand_dims(Hw_est_wahbas, axis=0)
	keypoints = keypoints_scaled.type(torch.int32)
	Pcs = keypoints_scaled_raw.type(torch.int32)

	# Hccc = batchPnP_Hc(Pws, keypoints_scaled,  K)
	# print("Hcccc : ",Hccc)

	return  predictions , keypoints , nameIDs, nameIDs_raw, cat_names, Hc_est, Hw_est,  Confidance_val, keep,  Pcs, Pc3ds, Pws, K, Hw_est_wno, W, Hw_est_wahbas
