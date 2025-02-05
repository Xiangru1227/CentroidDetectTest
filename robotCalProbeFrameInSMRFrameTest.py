import os
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

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

def getFrameTransform(centroids, iprobeParam:Prm, solidCubePrm:solidCube, trackerAngle:trackerAngle, tip1_pos, tip2_pos, tip3_pos, smr_pos):
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
    # print(f"fp: {fpinft}")
    
    smr_norm = fpinfb[:, 1]
    v_beam = np.array([0, 1, 0])
    temp = np.dot(v_beam, smr_norm)
    temp = clamp(temp, -1.0, 1.0)
    combo_angle = np.arccos(temp) * 180 / np.pi
    err_lat  = solidCubePrm.a_lat[0]  + solidCubePrm.a_lat[1]  * combo_angle + solidCubePrm.a_lat[2]  * combo_angle ** 2
    err_long = solidCubePrm.a_long[0] + solidCubePrm.a_long[1] * combo_angle + solidCubePrm.a_long[2] * combo_angle ** 2
    vTemp = np.cross(smr_norm, v_beam)
    vLatInFb = np.cross(v_beam, vTemp)
    compFb = err_lat * vLatInFb + err_long * np.array([0, -1, 0])
    compFt = np.dot(fbinft, compFb)
    
    fs_z = tip2_pos - tip3_pos
    fs_z = fs_z / np.linalg.norm(fs_z)
    fs_y = np.cross(-fs_z, tip1_pos - tip2_pos)
    fs_y = fs_y / np.linalg.norm(fs_y)
    fs_x = np.cross(fs_y, fs_z)
    fsinft = np.column_stack((fs_x, fs_y, fs_z))
    # print(f"fs: {fsinft}")
    
    smr_pos = smr_pos + compFt
    fsinft_inv = fsinft.T
    fpinfs = np.dot(fsinft_inv, fpinft)
    smr_in_fs = np.dot(fsinft_inv, (smr_pos - tip2_pos))
    
    fpinfs = np.dot(np.linalg.inv(fsinft), fpinft)
    
    return fpinfs, smr_in_fs, pyr

def process_file(i, iprobeParam, solidCubePrm, folder):
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

    return getFrameTransform(centroids, iprobeParam, solidCubePrm, tracker_angle, tip1_pos, tip2_pos, tip3_pos, smr_pos)

def mat_to_euler_std(fpinfs_list):
    euler_angles = np.array([R.from_matrix(mat).as_euler('xyz', degrees=True) for mat in fpinfs_list])
    std_angles = np.std(euler_angles, axis=0)
    # print(euler_angles)

    return euler_angles, std_angles

def mat_std(fpinfs_list):
    fpinfs_array = np.array(fpinfs_list)
    mean_matrix = np.mean(fpinfs_array, axis=0)
    std_matrix = np.std(fpinfs_array, axis=0)
    
    frob_stds = [np.linalg.norm(mat - mean_matrix, 'fro') for mat in fpinfs_list]
    frob_std = np.std(frob_stds)

    return std_matrix, frob_std

def pos_std(pos_list):
    return np.std(np.array(pos_list), axis=0)

def main():
    file_num = 18
    folder = 'iProbeCalibration/geoCalRobot3'
    # iprobe_prm_input = Prm(63.5, 77, -77)
    solidCube_prm_input = solidCube(0, 0, 0, 0, 0, 0)
    # iprobe_prm_input = Prm(65.39392092546034, 74.27754179794745, -78.90018349612558)
    # iprobe_prm_input = Prm(64.9431856199451, 74.32932306572513, -78.43056148009276)
    iprobe_prm_input = Prm(64.39860222953655, 75.37563614566022, -77.88563421698998)
    # iprobe_prm_input = Prm(65.09435929303649, 74.20233322262244, -78.42560752762888)
    # solidCube_prm_input = solidCube(0.4785964150348852, 0.3046415434970332, -0.003916539620741264, 
    #                                 5.949882180933179, -0.1232472204813352, 0.0029416898060632144)
    
    list_of_fpinfs = []
    list_of_smrinfs = []
    list_of_pyr = []
    for i in range(file_num):
        mat, pos, pyr = process_file(i, iprobe_prm_input, solidCube_prm_input, folder)
        list_of_fpinfs.append(mat)
        list_of_smrinfs.append(pos)
        list_of_pyr.append(pyr)
    
    euler_angles, std_angles = mat_to_euler_std(list_of_fpinfs)
    
    # std_matrix, frob_std = mat_std(list_of_fpinfs)
    # print("fp in fs std:")
    # print(std_matrix)
    # print("\nfp in fs Frobenius std:")
    # print(frob_std)

    print("Mean Euler Angles (degrees): [Pitch, Yaw, Roll]")
    print(np.mean(euler_angles, axis=0))

    print("\nEuler Angles Standard Deviation (degrees): [Pitch, Yaw, Roll]")
    print(std_angles)
     
    # smr_std = pos_std(list_of_smrinfs)
    # print("\nsmr in fs std:")
    # print(smr_std)
    
    plt.figure(figsize=(10, 25))

    plt.subplot(3, 2, 1)
    plt.scatter([pyr.pitch for pyr in list_of_pyr], euler_angles[:, 0], label="Pitch (probe in 3 SMR frame)", color="r")
    plt.scatter([pyr.pitch for pyr in list_of_pyr], euler_angles[:, 1], label="Yaw (probe in 3 SMR frame)", color="g")
    plt.scatter([pyr.pitch for pyr in list_of_pyr], euler_angles[:, 2], label="Roll (probe in 3 SMR frame)", color="b")
    plt.xlabel("Pitch (probe in tracker frame)")
    plt.ylabel("Euler Angles pitch (degrees)")
    plt.legend()
    # plt.title("Euler Angles vs. PYR")
    
    plt.subplot(3, 2, 2)
    plt.scatter([pyr.yaw for pyr in list_of_pyr], euler_angles[:, 0], label="Pitch (probe in 3 SMR frame)", color="r")
    plt.scatter([pyr.yaw for pyr in list_of_pyr], euler_angles[:, 1], label="Yaw (probe in 3 SMR frame)", color="g")
    plt.scatter([pyr.yaw for pyr in list_of_pyr], euler_angles[:, 2], label="Roll (probe in 3 SMR frame)", color="b")
    plt.xlabel("Yaw (probe in tracker frame)")
    plt.ylabel("Euler Angles yaw (degrees)")
    plt.legend()
    # plt.title("Euler Angles vs. PYR")
    
    plt.subplot(3, 2, 3)
    plt.scatter([pyr.roll for pyr in list_of_pyr], euler_angles[:, 0], label="Pitch (probe in 3 SMR frame)", color="r")
    plt.scatter([pyr.roll for pyr in list_of_pyr], euler_angles[:, 1], label="Yaw (probe in 3 SMR frame)", color="g")
    plt.scatter([pyr.roll for pyr in list_of_pyr], euler_angles[:, 2], label="Roll (probe in 3 SMR frame)", color="b")
    plt.xlabel("Roll (probe in tracker frame)")
    plt.ylabel("Euler Angles roll (degrees)")
    plt.legend()
    # plt.title("Euler Angles vs. PYR")
    
    plt.subplot(3, 2, 5)
    plt.scatter(np.arange(len(euler_angles)), euler_angles[:, 0], label="Pitch (probe in 3 SMR frame)", color="r")
    plt.scatter(np.arange(len(euler_angles)), euler_angles[:, 1], label="Yaw (probe in 3 SMR frame)", color="g")
    plt.scatter(np.arange(len(euler_angles)), euler_angles[:, 2], label="Roll (probe in 3 SMR frame)", color="b")
    plt.xlabel("Sequence")
    plt.ylabel("Euler Angles (degrees)")
    plt.legend()
    # plt.title("Euler Angles over Sequence")
    
    plt.subplot(3, 2, 6)
    plt.scatter(np.arange(len(euler_angles)), [pyr.pitch for pyr in list_of_pyr], label="Pitch (probe in tracker frame)", color="r")
    plt.scatter(np.arange(len(euler_angles)), [pyr.yaw for pyr in list_of_pyr], label="Yaw (probe in tracker frame)", color="g")
    plt.scatter(np.arange(len(euler_angles)), [pyr.roll for pyr in list_of_pyr], label="Roll (probe in tracker frame)", color="b")
    plt.xlabel("Sequence")
    plt.ylabel("Euler Angles (degrees)")
    plt.legend()
    # plt.title("Probe PYR over Sequence")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.subplots_adjust(bottom=0.1)
    plt.show()

if __name__ == '__main__':
    main()