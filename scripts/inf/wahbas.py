import numpy as np

def wahbasSimTransRot(H_Ibs, w, n = 200):
    print("H_Ibs size : ", H_Ibs.shape)
    print("w size : ", w.shape)
    '''
    H_Ibs is set of SE(3) which contain rotations (R's) and translations (t's).

    Reference
    [1] de Ruiter, A.H.J., Forbes, J.R. On the Solution ofWahba’s Problem on S O (n). J of Astronaut Sci 60, 1–31 (2013). https://doi.org/10.1007/s40295-014-0019-8
        https://link.springer.com/content/pdf/10.1007/s40295-014-0019-8.pdf

    '''
    # multiply n times to create many observations 
    w = np.tile(w, (n))
    H_Ibs = np.tile(H_Ibs, (n,1,1))

    # randomly generate observations from reference frame
    pIs = np.array([np.random.choice([-5,-4,-3,-2,-1,1,2,3,4,5], 3) for i in range(w.shape[0])])

    pbs = np.array([(H_Ibs[i][:,0:3].T @ np.expand_dims(pIs[i],axis=1) - H_Ibs[i][:,0:3].T @ np.expand_dims(H_Ibs[i][:,3].T,axis=1)).T[0] for i in range(w.shape[0])])

    wsum = np.sum(w)

    pI = np.sum(np.multiply(pIs.T, w).T, axis=0)/wsum

    pb = np.sum(np.multiply(pbs.T, w).T, axis=0)/wsum
    #p_II = wk(pIk - pI) 
    p_II = np.array([w[i]*(pIs[i]-pI) for i in range(w.shape[0])])
    #p_bb = pbk - pb)
    p_bb = np.array([(pbs[i]-pb) for i in range(w.shape[0])])

    Bts = np.array([np.expand_dims(p_II[i],axis=1) @ np.expand_dims(p_bb[i],axis=0) for i in range(w.shape[0])])
    Bt = np.sum(Bts,axis=0)

    U, S, Vt = np.linalg.svd(Bt)

    d = np.linalg.det(U)*np.linalg.det(Vt)
    M = np.diag([1, 1, d])
    R_est = U @ M @  Vt
    r_est = (-R_est @ np.expand_dims(pb,axis=1)).T[0] + pI
    H_est = np.concatenate((R_est,np.expand_dims(r_est,axis=1)),axis=1)
    
    return H_est