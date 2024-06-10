import torch
import torch.nn.functional as F

def pred_Line_CR2(P,i):
    if i < 3:
        L0,L1,L2,L3 = P[:,i,:],P[:,2*i+8,:],P[:,2*i+9,:],P[:,i+1,:]
    if i == 3:
        L0,L1,L2,L3 = P[:,i,:],P[:,2*i+8,:],P[:,2*i+9,:],P[:,i-3,:]
    if i > 3 and i < 8:
        L0,L1,L2,L3 = P[:,i-4,:],P[:,2*i+8,:],P[:,2*i+9,:],P[:,i,:]
    if i >= 8 and i < 11:
        L0,L1,L2,L3 = P[:,i-4,:],P[:,2*i+8,:],P[:,2*i+9,:],P[:,i-3,:]
    if i == 11:
        L0,L1,L2,L3 = P[:,i-4,:],P[:,2*i+8,:],P[:,2*i+9,:],P[:,i-7,:]

    AC=torch.diag(torch.inner(L2 - L0, L2 - L0),0) #c-a
    BC=torch.diag(torch.inner(L2 - L1, L2 - L1),0) #c-b
    BD=torch.diag(torch.inner(L3 - L1, L3 - L1),0) #d-b
    AD=torch.diag(torch.inner(L3 - L0, L3 - L0),0) #d-a

    return AC * BD / (BC * AD)

def pred_CR2(P_pred, device):
    #P_pred=P_pred.to(device, dtype=torch.float32)
    P = P_pred.to(device, dtype=torch.float32).view(-1,32,2)
    for i in range(12):
        cr_2 = pred_Line_CR2(P,i)
        if i == 0:
            CR_2 = cr_2
        else:
            CR_2 = torch.cat((CR_2, cr_2), dim = 0)
    return CR_2

def CR2_loss(P_pred, target_cr2, device):
    return F.smooth_l1_loss(pred_CR2(P_pred, device),target_cr2,reduction='sum')


## OLD Cross ratio implementation
def pointsOnALine(P_pred,edge):
    i = edge
    P = P_pred.reshape(32,2)
    if i < 3:
        L = torch.stack((P[i],P[2*i+8],P[2*i+9],P[i+1]))
    if i == 3:
        L = torch.stack((P[i],P[2*i+8],P[2*i+9],P[i-3]))
    if i > 3 and i < 8:
        L = torch.stack((P[i-4],P[2*i+8],P[2*i+9],P[i]))
    if i >= 8 and i < 11:
        L = torch.stack((P[i-4],P[2*i+8],P[2*i+9],P[i-3]))
    if i == 11:
        L = torch.stack((P[i-4],P[2*i+8],P[2*i+9],P[i-7]))
    return L

def appro_cr_sqrd_line(Line):
    """
    Approximate the square of cross-ratio along four ordered 2D points using 
    inner-product
    
    Line: PyTorch tensor of shape [4, 2]

    appro_cr_sqrd = (||c-a||^2 . ||d-b||^2) / (||c-b||^2.||d-a||^2)
    """
    AC = Line[2] - Line[0] #c-a
    BC = Line[2] - Line[1] #c-b
    BD = Line[3] - Line[1] #d-b
    AD = Line[3] - Line[0] #d-a

    return (AC.dot(AC) * BD.dot(BD)) / (BC.dot(BC) * AD.dot(AD)).unsqueeze(0)

def appro_cr_sqrd(P_pred_bs):
    for bs in range(P_pred_bs.size()[0]):
        for i in range(12):
            line = pointsOnALine(P_pred_bs[bs,:],i)
            a_cr = appro_cr_sqrd_line(line)
            if i == 0:
                A_cr = a_cr
            else:
                A_cr = torch.cat((A_cr, a_cr), dim = 0)
        A_cr = A_cr.unsqueeze(0)
        if bs == 0:
            Ap_cr = A_cr
        else:
            Ap_cr = torch.cat((Ap_cr, A_cr), dim = 0)
    return Ap_cr

def cr_loss(pred_coor, target_cr):
    approx_cr = appro_cr_sqrd(pred_coor)
    target_cr_sqrd = torch.ones(pred_coor.size()[0], 12).to(pred_coor.device)*target_cr**2
    loss = F.smooth_l1_loss(approx_cr,target_cr_sqrd,reduction='sum')
    return loss
## OLD