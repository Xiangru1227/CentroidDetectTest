import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

class Prm:
    def __init__(self, D, H1, H2):
        self.D = D
        self.H1 = H1
        self.H2 = H2
    
    def getPYR(self, centroids):
        centroids[2] = [np.mean([centroids[2][0], centroids[3][0]]), np.mean([centroids[2][1], centroids[3][1]])]
        
        u1 = centroids[0][0] - centroids[1][0]
        v1 = -(centroids[0][1] - centroids[1][1])
        u2 = centroids[2][0] - centroids[1][0]
        v2 = -(centroids[2][1] - centroids[1][1])
        # print("u1v1: ", u1, v1)
        # print("u2v2: ", u2, v2)

        roll = (math.atan2((u1 - u2), (v1 - v2))) * 180 / np.pi
        sr = math.sin(roll * np.pi / 180)
        cr = math.cos(roll * np.pi / 180)

        m = math.sqrt(((u1 - u2) ** 2 + (v1 - v2) ** 2) / (self.H1 - self.H2) ** 2)
        n = (sr * v1 - cr * u1) / self.D
        k = (sr * u1 + cr * v1 - self.H1 * m) / self.D

        if abs(n) < 0.00001:
            temp1 = m ** 2 + n ** 2 + k ** 2
            temp2 = m ** 2 * n ** 2
            ss = (temp1 - math.sqrt(temp1 ** 2 - 4 * temp2)) / (2 * temp2)
            scale = math.sqrt(ss)
        else:
            scale = 1 / math.sqrt(m ** 2 + k ** 2)

        yaw = (math.asin(clamp(n * scale, -1.0, 1.0))) * 180 / np.pi
        # print(k * scale / math.cos((yaw / 180) * np.pi))
        pitch = (math.asin(clamp(k * scale / math.cos((yaw / 180) * np.pi), -1.0, 1.0))) * 180 / np.pi

        return PYR(pitch, yaw, roll)
    
class solidCube:
    def __init__(self, a0_lat, a1_lat, a2_lat, a0_long, a1_long, a2_long):
        self.a_lat = [a0_lat, a1_lat, a2_lat]
        self.a_long = [a0_long, a1_long, a2_long]
        
class PYR:
    def __init__(self, pitch, yaw, roll):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll
        
class trackerAngle:
    def __init__(self, az, el):
        self.az = az
        self.el = el
    
    def getFbInFt(self):
        rz = R.from_euler('z', self.az, degrees=True).as_matrix()
        rx = R.from_euler('x', self.el, degrees=True).as_matrix()
        
        return np.dot(rz, rx)

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

'''cal with long stylus'''
def getTipOffset(centroids, iprobeParam:Prm, solidCubePrm:solidCube, trackerAngle:trackerAngle, tip_pos, smr_pos):
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
    # ftinfp = np.linalg.inv(fpinft)
    
    smr_norm = fpinfb[:, 1]
    # smr_norm = fbinft[:, 1]
    v_beam = np.array([0, 1, 0])
    
    temp = np.dot(v_beam, smr_norm)
    temp = clamp(temp, -1.0, 1.0)
    combo_angle = np.arccos(temp) * 180 / np.pi
    # print(combo_angle)
    
    err_lat  = solidCubePrm.a_lat[0]  + solidCubePrm.a_lat[1]  * combo_angle + solidCubePrm.a_lat[2]  * combo_angle ** 2
    err_long = solidCubePrm.a_long[0] + solidCubePrm.a_long[1] * combo_angle + solidCubePrm.a_long[2] * combo_angle ** 2
    
    vTemp = np.cross(smr_norm, v_beam)
    vLatInFb = np.cross(v_beam, vTemp)
    
    compFb = err_lat * vLatInFb + err_long * np.array([0, -1, 0])
    compFt = np.dot(fbinft, compFb)
    tipOffset = tip_pos - smr_pos
    
    return np.linalg.inv(fpinft) @ np.transpose(tipOffset - compFt)

def process_file(i, iprobeParam, solidCubePrm, folder):
    file_path = str(i+1) + '.txt'
    with open(os.path.join(folder, file_path), 'r') as f:
        lines = f.readlines()
    centroids = np.array([float(x) for x in lines[0].split()]).reshape(4, 2)
    smr_pos = np.array([float(x) for x in lines[1].split()])
    tracker_az, tracker_el = [float(x) for x in lines[2].split()]
    tip_pos = np.array([float(x) for x in lines[3].split()])
    
    # tip_offset_in_ft = tip_pos - smr_pos
    tracker_angle = trackerAngle(tracker_az, tracker_el)

    return getTipOffset(centroids, iprobeParam, solidCubePrm, tracker_angle, tip_pos, smr_pos)

def getError(iprobeParam, solidCubePrm, mode, idx, folder):
    tip_offset_in_fp_list = []
    for i in range(idx):
        tip_offset_in_fp_list.append(process_file(i, iprobeParam, solidCubePrm, folder))
        
    # for tip in tip_offset_in_fp_list:
    #     print(f"[{tip[0]:.4f},\t{tip[1]:.4f},\t{tip[2]:.4f}]")
    # tip_offset_avg = np.mean(tip_offset_in_fp_list, axis=0)
    # print(f"Average tip offset: [{tip_offset_avg[0]:.4f},\t{tip_offset_avg[1]:.4f},\t{tip_offset_avg[2]:.4f}]")
    
    offset_x_std = np.std([offset[0] for offset in tip_offset_in_fp_list])
    offset_y_std = np.std([offset[1] for offset in tip_offset_in_fp_list])
    offset_z_std = np.std([offset[2] for offset in tip_offset_in_fp_list])
    
    if mode == 'scalar':
        return offset_x_std ** 2 + offset_y_std ** 2 + offset_z_std ** 2
    elif mode == 'vector':
        return np.array([offset_x_std, offset_y_std, offset_z_std])
    elif mode == 'vector+':
        residuals = np.array([offset_x_std, offset_y_std, offset_z_std])
        
        num_params = len(solidCubePrm.a_lat) + len(solidCubePrm.a_long)
        if len(residuals) < num_params:
            additional_residuals = np.zeros(num_params - len(residuals))
            residuals = np.concatenate((residuals, additional_residuals))
        
        return residuals
    else:
        raise ValueError("Invalid mode.")

'''cal with robot'''
def getTipOffset_robot(centroids, iprobeParam:Prm, solidCubePrm:solidCube, trackerAngle:trackerAngle, tip1_pos, tip2_pos, tip3_pos, smr_pos):
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
    # ftinfp = np.linalg.inv(fpinft)
    
    smr_norm = fpinfb[:, 1]
    # smr_norm = fbinft[:, 1]
    v_beam = np.array([0, 1, 0])
    
    temp = np.dot(v_beam, smr_norm)
    temp = clamp(temp, -1.0, 1.0)
    combo_angle = np.arccos(temp) * 180 / np.pi
    # print(combo_angle)
    
    err_lat  = solidCubePrm.a_lat[0]  + solidCubePrm.a_lat[1]  * combo_angle + solidCubePrm.a_lat[2]  * combo_angle ** 2
    err_long = solidCubePrm.a_long[0] + solidCubePrm.a_long[1] * combo_angle + solidCubePrm.a_long[2] * combo_angle ** 2
    
    vTemp = np.cross(smr_norm, v_beam)
    vLatInFb = np.cross(v_beam, vTemp)
    
    compFb = err_lat * vLatInFb + err_long * np.array([0, -1, 0])
    compFt = np.dot(fbinft, compFb)
    
    tipOffset1 = tip1_pos - smr_pos
    tipOffset2 = tip2_pos - smr_pos
    tipOffset3 = tip3_pos - smr_pos
    tipOffset1_fp = np.linalg.inv(fpinft) @ np.transpose(tipOffset1 - compFt)
    tipOffset2_fp = np.linalg.inv(fpinft) @ np.transpose(tipOffset2 - compFt)
    tipOffset3_fp = np.linalg.inv(fpinft) @ np.transpose(tipOffset3 - compFt)
    
    return tipOffset1_fp, tipOffset2_fp, tipOffset3_fp

def process_file_robot(i, iprobeParam, solidCubePrm, folder):
    file_path = str(i+1) + '.txt'
    with open(os.path.join(folder, file_path), 'r') as f:
        lines = f.readlines()
    tracker_az, tracker_el = [float(x) for x in lines[0].split()]
    smr_pos = np.array([float(x) for x in lines[1].split()])
    centroids = np.array([float(x) for x in lines[2].split()]).reshape(4, 2)
    tip1_pos = np.array([float(x) for x in lines[3].split()])
    tip2_pos = np.array([float(x) for x in lines[4].split()])
    tip3_pos = np.array([float(x) for x in lines[5].split()])
    '''1 is top-left, 2 is top-right, 3 is bottom right'''
    
    # tip_offset_in_ft = tip_pos - smr_pos
    tracker_angle = trackerAngle(tracker_az, tracker_el)

    return getTipOffset_robot(centroids, iprobeParam, solidCubePrm, tracker_angle, tip1_pos, tip2_pos, tip3_pos, smr_pos)

def getError_robot(iprobeParam, solidCubePrm, mode, idx, folder):
    tipOffset1_fp_list = []
    tipOffset2_fp_list = []
    tipOffset3_fp_list = []
    for i in range(idx):
        tipOffset1, tipOffset2, tipOffset3 = process_file_robot(i, iprobeParam, solidCubePrm, folder)
        tipOffset1_fp_list.append(tipOffset1)
        tipOffset2_fp_list.append(tipOffset2)
        tipOffset3_fp_list.append(tipOffset3)
    
    # for tip in tipOffset1_fp_list:
    #     print(f"[{tip[0]:.4f},\t{tip[1]:.4f},\t{tip[2]:.4f}]")
    # tip_offset_avg = np.mean(tipOffset1_fp_list, axis=0)
    # print(f"Average tip offset: [{tip_offset_avg[0]:.4f},\t{tip_offset_avg[1]:.4f},\t{tip_offset_avg[2]:.4f}]\n")
    
    # for tip in tipOffset2_fp_list:
    #     print(f"[{tip[0]:.4f},\t{tip[1]:.4f},\t{tip[2]:.4f}]")
    # tip_offset_avg = np.mean(tipOffset2_fp_list, axis=0)
    # print(f"Average tip offset: [{tip_offset_avg[0]:.4f},\t{tip_offset_avg[1]:.4f},\t{tip_offset_avg[2]:.4f}]\n")
    
    # for tip in tipOffset3_fp_list:
    #     print(f"[{tip[0]:.4f},\t{tip[1]:.4f},\t{tip[2]:.4f}]")
    # tip_offset_avg = np.mean(tipOffset3_fp_list, axis=0)
    # print(f"Average tip offset: [{tip_offset_avg[0]:.4f},\t{tip_offset_avg[1]:.4f},\t{tip_offset_avg[2]:.4f}]\n")
    
    offset1_std = np.std(tipOffset1_fp_list, axis=0)
    offset2_std = np.std(tipOffset2_fp_list, axis=0)
    offset3_std = np.std(tipOffset3_fp_list, axis=0)
    
    if mode == 'scalar':
        return np.sqrt(np.sum(offset1_std ** 2 + offset2_std ** 2 + offset3_std ** 2))
    elif mode == 'vector':
        return np.concatenate([offset1_std, offset2_std, offset3_std])
    else:
        raise ValueError("Invalid mode.")

def getDiff_robot(iprobeParam, solidCubePrm, idx, folder):
    tipOffset1_fp_list = []
    tipOffset2_fp_list = []
    tipOffset3_fp_list = []
    pyr_list = []
    
    for i in range(idx):
        tipOffset1, tipOffset2, tipOffset3 = process_file_robot(i, iprobeParam, solidCubePrm, folder)
        centroids = np.loadtxt(os.path.join(folder, f"{i+1}.txt"), skiprows=2, max_rows=1).reshape(4, 2)  # 读取 `centroids`
        pyr = iprobeParam.getPYR(centroids)

        tipOffset1_fp_list.append(tipOffset1)
        tipOffset2_fp_list.append(tipOffset2)
        tipOffset3_fp_list.append(tipOffset3)
        pyr_list.append([pyr.pitch, pyr.yaw, pyr.roll])
    
    tipRef1 = tipOffset1_fp_list[0]
    tipRef2 = tipOffset2_fp_list[0]
    tipRef3 = tipOffset3_fp_list[0]
    
    tipDiff1 = np.linalg.norm(np.array(tipOffset1_fp_list) - tipRef1, axis=1)
    tipDiff2 = np.linalg.norm(np.array(tipOffset2_fp_list) - tipRef2, axis=1)
    tipDiff3 = np.linalg.norm(np.array(tipOffset3_fp_list) - tipRef3, axis=1)

    pyr_array = np.array(pyr_list)

    labels = ['Pitch (°)', 'Yaw (°)', 'Roll (°)']
    tip_labels = ['Tip 1', 'Tip 2', 'Tip 3']
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    
    roll_indices = np.arange(1, 6)
    yaw_indices = np.arange(6, 12)
    pitch_indices = np.arange(12, 18)
    for tip_idx, tipDiff in enumerate([tipDiff1, tipDiff2, tipDiff3]):
        for pyr_idx, indices in enumerate([pitch_indices, yaw_indices, roll_indices]):
            ax = axes[tip_idx, pyr_idx]
            ax.scatter(pyr_array[indices, pyr_idx], tipDiff[indices], alpha=0.7)
            ax.set_xlabel(labels[pyr_idx])
            ax.set_ylabel("Offset Difference (mm)")
            ax.set_title(f"{tip_labels[tip_idx]} vs {labels[pyr_idx]} (Group {indices[0]+1}-{indices[-1]+1})")
    
    # for tip_idx, tipDiff in enumerate([tipDiff1, tipDiff2, tipDiff3]):
    #     for pyr_idx in range(3):
    #         ax = axes[tip_idx, pyr_idx]
    #         ax.scatter(pyr_array[:, pyr_idx], tipDiff, alpha=0.7)
    #         ax.set_xlabel(labels[pyr_idx])
    #         ax.set_ylabel("Offset Difference (mm)")
    #         ax.set_title(f"{tip_labels[tip_idx]} vs {labels[pyr_idx]}")
    
    plt.tight_layout()
    plt.show()