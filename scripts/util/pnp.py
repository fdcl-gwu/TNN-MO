import torch
import torch.nn.functional as F
from pytorch3d.ops import efficient_pnp


def Epnp_T_est_loss(pred_keypoints , targets_keypoints, targets_keypoints3dw, Denorm, K_inv):
    traget_transform = efficient_pnp(targets_keypoints3dw.view(-1, 32, 3), 
                                    torch.matmul(torch.cat(((targets_keypoints*Denorm[:targets_keypoints.size()[0]]).view(-1, 32, 2), 
                                                 torch.ones(((targets_keypoints*Denorm[:targets_keypoints.size()[0]]).view(-1, 32, 2)).size()[0], 32, 1).to(targets_keypoints.device)), dim=-1), 
                                    K_inv)[...,:2])
    pred_transform = efficient_pnp(targets_keypoints3dw.view(-1, 32, 3), 
                                   torch.matmul(torch.cat(((pred_keypoints*Denorm[:pred_keypoints.size()[0]]).view(-1, 32, 2), 
                                                torch.ones(((pred_keypoints*Denorm[:pred_keypoints.size()[0]]).view(-1, 32, 2)).size()[0], 32, 1).to(pred_keypoints.device)), dim=-1), 
                                   K_inv)[...,:2])

    return F.l1_loss(pred_transform.T, traget_transform.T, reduction='sum')

def pnp_T_est_loss(pred_keypoints , targets_keypoints, targets_keypoints3dw, Denorm, K_inv, pc_3d_vec, ):
    keypoints_3dw = targets_keypoints3dw.view(-1, 32, 3)

    targets_keypoints = targets_keypoints*Denorm[:targets_keypoints.size()[0]] #Denormalized
    pred_keypoints = pred_keypoints*Denorm[:targets_keypoints.size()[0]] #Denormalized
    #target_key_xy = target_keypoint_xy_Denorm #to(dtype=torch.int32)
    #target_keypoint_xy = 

    pc_3d_vec[:targets_keypoints.size()[0],:,:2] = targets_keypoints.view(-1, 32, 2)

    Pc_2d_uncal = torch.matmul(pc_3d_vec[:targets_keypoints.size()[0],:,:], K_inv)
    traget_transform = efficient_pnp(keypoints_3dw, Pc_2d_uncal[...,:2])

    # pred_key_xy = pred_keypoint_xy_Denorm #to(dtype=torch.int32)
    # pred_keypoint_xy = pred_keypoints.view(-1, 32, 2)
    pc_3d_vec[:targets_keypoints.size()[0],:,:2] = pred_keypoints.view(-1, 32, 2)

    Pc_2d_uncal = torch.matmul(pc_3d_vec[:targets_keypoints.size()[0],:,:], K_inv)
    pred_transform = efficient_pnp(keypoints_3dw, Pc_2d_uncal[...,:2])
    
    return F.l1_loss(pred_transform.T, traget_transform.T, reduction='sum')


