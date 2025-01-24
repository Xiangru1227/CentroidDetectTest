import pandas as pd
from cal_utils import *

probeData = pd.read_excel('geoCal2/raw/probe1.xlsx')
# print(probeData.columns)

tipData = pd.read_excel('geoCal2/raw/tip1.xlsx')

iprobePrm = Prm(66.93134649, 74.17418249, -77.03010405)
solidCubePrm = solidCube(9.04244247e-02, 1.04356156e+00, -3.25926171e-02, 
                         5.56077254e+00,3.60858917e-02, 9.99922056e-05)
tracker_angle = []
smr_pos = []
centroids = []
pyrs = []
tips = []

# probeAvg = probeData.iloc[-1]
# trkAngleAvg = trackerAngle(probeAvg['AZ'], probeAvg['EL'])
# smrPosAvg = [probeAvg['X'], probeAvg['Y'], probeAvg['Z']]
# centroidsAvg = [[probeAvg['C1x'], probeAvg['C1y']], 
#                 [probeAvg['C2x'], probeAvg['C2y']], 
#                 [probeAvg['C3x'], probeAvg['C3y']], 
#                 [probeAvg['C4x'], probeAvg['C4y']]]
# pyrAvg = iprobePrm.getPYR(centroidsAvg)
# ry = R.from_euler('y', pyrAvg.roll,  degrees=True).as_matrix()
# rx = R.from_euler('x', pyrAvg.pitch, degrees=True).as_matrix()
# rz = R.from_euler('z', pyrAvg.yaw,   degrees=True).as_matrix()
# fpinfcAvg = np.dot(np.dot(ry, rx), rz)

# fbinftAvg = trkAngleAvg.getFbInFt()
# fcinfbAvg = R.from_euler('y', 0.16, degrees=True).as_matrix()
# fpinfbAvg = np.dot(fcinfbAvg, fpinfcAvg)
# fpinftAvg = np.dot(fbinftAvg, fpinfbAvg)

# tipAvg = tipData.iloc[-1]
# tipPos = np.array([tipAvg['X'], tipAvg['Y'], tipAvg['Z']])

# smr_normAvg = fpinfbAvg[:, 1]
# v_beamAvg = np.array([0, 1, 0])

# tempAvg = np.dot(v_beamAvg, smr_normAvg)
# tempAvg = clamp(tempAvg, -1.0, 1.0)
# combo_angleAvg = np.arccos(tempAvg) * 180 / np.pi

# err_latAvg  = solidCubePrm.a_lat[0]  + solidCubePrm.a_lat[1]  * combo_angleAvg + solidCubePrm.a_lat[2]  * combo_angleAvg ** 2
# err_longAvg = solidCubePrm.a_long[0] + solidCubePrm.a_long[1] * combo_angleAvg + solidCubePrm.a_long[2] * combo_angleAvg ** 2

# vTempAvg = np.cross(smr_normAvg, v_beamAvg)
# vLatInFbAvg = np.cross(v_beamAvg, vTempAvg)

# compFbAvg = err_latAvg * vLatInFbAvg + err_longAvg * np.array([0, -1, 0])
# compFtAvg = np.dot(fbinftAvg, compFbAvg)

# tipOffset = np.linalg.inv(fpinftAvg) @ np.transpose(np.array(tipPos) - np.array(smrPosAvg) - np.array(compFtAvg))
# print(f"tip offset: {tipOffset}")

tipOffset = [0.2256, 3.3326, -246.6527]

for _, row in probeData.iloc[1:-1].iterrows():
    tracker_angle.append(trackerAngle(row['AZ'], row['EL']))
    smr_pos.append([row['X'], row['Y'], row['Z']])
    centroids.append([[row['C1x'], row['C1y']], 
                      [row['C2x'], row['C2y']], 
                      [row['C3x'], row['C3y']], 
                      [row['C4x'], row['C4y']]])
    
# print(tracker_angle[0].getFbInFt())
for i in range(len(centroids)):
    pyr = iprobePrm.getPYR(centroids[i])
    pyrs.append(pyr)
    ry = R.from_euler('y', pyr.roll,  degrees=True).as_matrix()
    rx = R.from_euler('x', pyr.pitch, degrees=True).as_matrix()
    rz = R.from_euler('z', pyr.yaw,   degrees=True).as_matrix()
    fpinfc = np.dot(np.dot(ry, rx), rz)
    
    fbinft = tracker_angle[i].getFbInFt()
    fcinfb = R.from_euler('y', 0.16, degrees=True).as_matrix()
    fpinfb = np.dot(fcinfb, fpinfc)
    fpinft = np.dot(fbinft, fpinfb)
    
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
    
    tipOffsetInFt = np.dot(fpinft, tipOffset)
    tips.append(smr_pos[i] + compFt + tipOffsetInFt)

mean = np.mean(tips, axis=0)
print(f"Mean: ({mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f})")
std = np.std(tips, axis=0)
print(f"Std: ({std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f})")
# meanErr = np.mean(tips - tipPos, axis=0)
# print(f"Mean error: ({meanErr[0]:.6f}, {meanErr[1]:.6f}, {meanErr[2]:.6f})")
range = np.ptp(tips, axis=0)
print(f"Range: ({range[0]:.4f}, {range[1]:.4f}, {range[2]:.4f})")
print(f"PYR rep: ({np.std([pyr.pitch for pyr in pyrs]):.6f}, {np.std([pyr.yaw for pyr in pyrs]):.6f}, {np.std([pyr.roll for pyr in pyrs]):.6f})")