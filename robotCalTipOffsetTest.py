from calUtils import *

iprobe_prm_input = Prm(65.09435929303649, 74.20233322262244, -78.42560752762888)
solidCube_prm_input = solidCube(0.4785964150348852, 0.3046415434970332, -0.003916539620741264, 
                                    5.949882180933179, -0.1232472204813352, 0.0029416898060632144)

getDiff_robot(iprobe_prm_input, solidCube_prm_input, 18, 'iProbeCalibration/geoCalRobot2')