import numpy as np
from scipy.linalg import svd
import plotly.graph_objects as go
from scipy.stats import norm
import torch
from scipy.linalg import logm

def bayesian_fusion_old(H_ests, ws):
    print("H_ests ", H_ests.shape)
    print("weights ", ws.shape)
    print("weights ", ws)
    print("sum weights  ", np.sum(ws))
    # Normalize the weights so they sum up to 1
    # ws /= np.sum(ws)

    # Calculate the weighted average of the rotation matrices and translation vectors
    A = sum(w * H_est[:3, :3] for w, H_est in zip(ws, H_ests))
    b = sum(w * H_est[:3, 3] for w, H_est in zip(ws, H_ests))

    # Compute the SVD of A
    U, s, Vt = svd(A)

    # The optimal rotation matrix R is given by R = UV^T
    R = np.dot(U, Vt)

    # However, to ensure R is a proper rotation matrix (i.e., its determinant is 1),
    # we introduce a correction matrix J
    J = np.eye(3)
    J[2, 2] = np.linalg.det(np.dot(U, Vt))

    # So the final optimal rotation matrix is R = UJV^T
    R = np.dot(U, np.dot(J, Vt))

    # The optimal translation vector t is simply the mean position b
    t = b

    # Combine R and t into a single 3x4 matrix H_optimal
    H_optimal = np.hstack((R, t.reshape(-1, 1)))

    return H_optimal


def bayesian_fusion(H_ests, w):

    # Extract rotation matrices and translation vectors from H_ests
    Rs = H_ests[:, :3, :3].cpu().numpy()
    ts = H_ests[:, :3, 3].cpu().numpy()
 
    # Compute weighted sum of the translation vectors
    E_t =  np.sum(w[:, None] * ts, axis=0)
    # w_reshaped = np.expand_dims(w, axis=-1) # Reshape w for broadcasting
    # E_t = np.sum(w_reshaped * ts, axis=0)

    # Compute covariance of the translation vectors
    Cov_P =   torch.from_numpy(np.sum(w[:, None, None] * (ts - E_t)[:, :, None] @ (ts - E_t)[:, None, :], axis=0))
    # diff = ts - E_t
    # cov_t = np.sum(w_reshaped * np.matmul(np.expand_dims(diff, axis=-1), np.expand_dims(diff, axis=-2)), axis=0)

    # Compute standard deviation vector for position
    position_std_dev = torch.from_numpy(np.sqrt(np.diag(Cov_P)))

    # Compute weighted sum of the rotation matrices
    E_R = np.sum(w[:, None, None] * Rs, axis=0)

    # Perform SVD on E_R
    U, D, V = properSVD(E_R)

    # Compute the maximum likelihood estimate of the attitude
    R_est = U @ D @ V.T

    # Check the determinant of R_est
    det_R_est = np.linalg.det(R_est)

    if det_R_est < 0:
        print("The determinant of R_est is -", det_R_est)

    # Combine E[R] and E[t] into H_mean
    H_mean = np.concatenate((R_est, np.expand_dims(E_t, axis=1)), axis=1)
    
    # Compute Sigma
    d1, d2, d3 = np.diag(D)
    Sigma = np.array([1 + d1 - d2 - d3, 1 - d1 + d2 - d3, 1 - d1 - d2 + d3])
    Cov_R =  torch.from_numpy(np.diag(Sigma))

     # Compute standard deviation vector for position
    attitude_std_dev = torch.from_numpy(np.sqrt(np.abs(Sigma)))
    #
    U = torch.from_numpy(U)
    D = torch.diag(torch.from_numpy(D))
    V = torch.from_numpy(V)

    return H_mean, Cov_P, position_std_dev, Cov_R, attitude_std_dev, U, D, V

def properSVD(R):
    # Perform SVD on R
    U_prime, D_prime, V_prime_T = np.linalg.svd(R)

    # Compute U, D, V for the proper SVD of R
    U = U_prime @ np.diag([1, 1, np.linalg.det(U_prime)])
    D = np.diag(D_prime) @ np.diag([1, 1, np.linalg.det(U_prime @ V_prime_T.T)])
    V = V_prime_T.T @ np.diag([1, 1, np.linalg.det(V_prime_T)])

    return U, D, V

def attitude_error_eta(U, V, R_d):
    # R_d is 3x3 rotation matrix representing the actual and desired attitudes
    # U is 3x3 is an orthogonal matrix from proper single value dicomposition
    # V is 3x3 is an orthogonal matrix from proper single value dicomposition
    # $$\eta = \logm \(U^T R_d V \)^\vee $$ 
    # Compute the matrix logarithm of the product U^T * R_d * V
    logm_result = logm(U.T @ R_d @ V)

    # The vee operator maps a 3x3 skew-symmetric matrix to a 3x1 vector
    eta = np.array([logm_result[2, 1], logm_result[0, 2], logm_result[1, 0]])
    return eta
 
# def bayesian_fusion(H_ests, w):
#     # Extract rotation matrices and translation vectors from H_ests
#     Rs = H_ests[:, :3, :3]
#     ts = H_ests[:, :3, 3]

#     # Reshape w for broadcasting
#     w_reshaped = w.unsqueeze(-1)

#     # Compute weighted sum of the translation vectors
#     E_t = torch.sum(w_reshaped * ts, dim=0)

#     # Compute covariance of the translation vectors
#     diff = ts - E_t
#     cov_t = torch.sum(w_reshaped * diff.unsqueeze(-1) @ diff.unsqueeze(-2), dim=0)

#     # Compute standard deviation vector
#     std_dev = torch.sqrt(torch.diag(cov_t))

#     # Compute weighted sum of the rotation matrices
#     E_R = torch.sum(w_reshaped * Rs, dim=0)

#     # Perform SVD on E_R
#     U_prime, D_prime, V_prime_T = torch.linalg.svd(E_R)

#     # Compute U, D, V for the proper SVD of E_R
#     U = U_prime @ torch.diag(torch.tensor([1, 1, torch.linalg.det(U_prime)]))
#     D = D_prime @ torch.diag(torch.tensor([1, 1, torch.linalg.det(U_prime @ V_prime_T.T)]))
#     V = V_prime_T.T @ torch.diag(torch.tensor([1, 1, torch.linalg.det(V_prime_T)]))

#     # Compute the maximum likelihood estimate of the attitude
#     R_est = U @ V.T

#     # Check the determinant of R_est
#     det_R_est = torch.linalg.det(R_est)
#     if det_R_est > 0:
#         print("The determinant of R_est is +", det_R_est.item())
#     elif det_R_est < 0:
#         print("The determinant of R_est is -", det_R_est.item())
#     else:
#         print("The determinant of R_est is 0.")

#     # Combine E[R] and E[t] into H_mean
#     H_mean = torch.cat((R_est, E_t.unsqueeze(-1)), dim=1)
    
#     # Compute Sigma
#     d1, d2, d3 = torch.diag(D)
#     Sigma = torch.diag(torch.tensor([1 + d1 - d2 - d3, 1 - d1 + d2 - d3, 1 - d1 - d2 + d3]))

#     return H_mean.cpu().numpy(), cov_t, std_dev, Sigma
