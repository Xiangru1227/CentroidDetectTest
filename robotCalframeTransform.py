from calUtils import Prm, solidCube, PYR, trackerAngle
import os
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize, least_squares

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

def getFrameTransform(centroids, iprobeParam:Prm, trackerAngle:trackerAngle, tip1_pos, tip2_pos, tip3_pos):
    pyr = iprobeParam.getPYR(centroids)
    # print(f"{pyr.pitch},\t{pyr.roll},\t{pyr.yaw}")
    
    ry = R.from_euler('y', pyr.roll,  degrees=True).as_matrix()
    rx = R.from_euler('x', pyr.pitch, degrees=True).as_matrix()
    rz = R.from_euler('z', pyr.yaw,   degrees=True).as_matrix()
    fpinfc = np.dot(np.dot(ry, rx), rz)
    
    fbinft = trackerAngle.getFbInFt()
    fcinfb = R.from_euler('y', 0.16, degrees=True).as_matrix()
    fpinfb = np.dot(fcinfb, fpinfc)
    fpinft = np.dot(fbinft, fpinfb)
    
    fs_z = tip2_pos - tip3_pos
    fs_z = fs_z / np.linalg.norm(fs_z)
    fs_y = np.cross(-fs_z, tip1_pos - tip2_pos)
    fs_y = fs_y / np.linalg.norm(fs_y)
    fs_x = np.cross(fs_y, fs_z)
    fsinft = np.column_stack((fs_x, fs_y, fs_z))
    fpinfs = np.dot(np.linalg.inv(fsinft), fpinft)
    
    return fpinfs

def process_file(i, iprobeParam, folder):
    file_path = str(i+1) + '.txt'
    with open(os.path.join(folder, file_path), 'r') as f:
        lines = f.readlines()
    tracker_az, tracker_el = [float(x) for x in lines[0].split()]
    smr_pos = np.array([float(x) for x in lines[1].split()])
    centroids = np.array([float(x) for x in lines[2].split()]).reshape(4, 2)
    tip1_pos = np.array([float(x) for x in lines[3].split()])
    tip2_pos = np.array([float(x) for x in lines[4].split()])
    tip3_pos = np.array([float(x) for x in lines[5].split()])
    tracker_angle = trackerAngle(tracker_az, tracker_el)

    return getFrameTransform(centroids, iprobeParam, tracker_angle, tip1_pos, tip2_pos, tip3_pos)

def mat_to_euler(fpinfs_list):
    return np.array([R.from_matrix(mat).as_euler('xyz', degrees=True) for mat in fpinfs_list])

def getError(iprobePrm):
    file_num = 18
    folder = 'iProbeCalibration/geoCalRobot3'
    iprobe_prm_input = Prm(iprobePrm[0], iprobePrm[1], iprobePrm[2])
    
    list_of_fpinfs = []
    for i in range(file_num):
        mat= process_file(i, iprobe_prm_input, folder)
        list_of_fpinfs.append(mat)
    
    euler_angles = mat_to_euler(list_of_fpinfs)
    return np.std(euler_angles, axis=0)

def prm_opt(iprobe_prm_input, tol=1e-6, max_iter=1000):
    prev_error = np.inf
    prev_iprobe_prm = iprobe_prm_input
    print(f"Error befor cal: {np.linalg.norm(getError(iprobe_prm_input))}")

    for iteration in range(max_iter):
        result_iprobe = minimize(fun=lambda x: np.linalg.norm(getError(x)), 
                                 x0=iprobe_prm_input, 
                                 method='L-BFGS-B', 
                                 bounds=[(53.5, 73.5), (67.0, 87.0), (-87.0, -67.0)], 
                                 tol=tol,
                                 options={'maxiter': max_iter, 'disp': False})
        iprobe_prm_input = result_iprobe.x
        
        current_error = np.linalg.norm(getError(iprobe_prm_input))
        print(f"Iteration {iteration+1}, Error: {current_error:.8f}")
        
        if abs(prev_error - current_error) < tol:
            print("Converged.")
            break
        elif current_error >= prev_error:
            print("Converged due to increasing error.")
            return prev_iprobe_prm, prev_error

        prev_error = current_error
        prev_iprobe_prm = iprobe_prm_input

    return iprobe_prm_input, current_error
    
def main():
    # iprobe_prm_input = [63.5, 77, -77]
    iprobe_prm_input = [64.9431856199451, 74.32932306572513, -78.43056148009276]
    final_iprobe_prm, final_error = prm_opt(iprobe_prm_input)
    
    print("\nFinal optimized parameters:")
    print(f"iProbe Parameters: [{final_iprobe_prm[0]}, {final_iprobe_prm[1]}, {final_iprobe_prm[2]}]")
    print("Final Error:", final_error)

if __name__ == '__main__':
    main()