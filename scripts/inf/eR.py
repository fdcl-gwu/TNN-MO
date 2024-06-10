import numpy as np

def attitude_error(R, R_d):
  # R and R_d are 3x3 rotation matrices representing the actual and desired attitudes
  # Returns a 3-element vector representing the attitude error
  e_R = 0.5 * (R_d.T @ R - R.T @ R_d) # Matrix multiplication
  e_R = np.array([e_R[2, 1], e_R[0, 2], e_R[1, 0]]) # Apply the vee map
  return e_R

def attitude_error_sequence(R_seq, R_d_seq):
    # R_seq and R_d_seq are lists of 3x3 rotation matrices
    # Returns a list of 3-element vectors representing the attitude errors
    assert len(R_seq) == len(R_d_seq)
    errors = [attitude_error(R, R_d) for R, R_d in zip(R_seq, R_d_seq)]
    return errors

def mean_abs_attitude_error(R_seq, R_d_seq):
    # R_seq and R_d_seq are lists of 3x3 rotation matrices
    # Returns the mean absolute attitude error
    assert len(R_seq) == len(R_d_seq)
    errors = [attitude_error(R, R_d) for R, R_d in zip(R_seq, R_d_seq)]
    mean_abs_error = np.mean([np.abs(e) for e in errors])
    return mean_abs_error