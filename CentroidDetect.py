import cv2
import numpy as np
import math

class IPPrm:
    def __init__(self, D, H1, H2):
        self.D = D
        self.H1 = H1
        self.H2 = H2

def api_ipb_3uv2pyr(keypoints, pyr, prm):
    u1 = keypoints[2].pt[0] - keypoints[0].pt[0]
    v1 = -(keypoints[2].pt[1] - keypoints[0].pt[1])
    u2 = keypoints[1].pt[0] - keypoints[0].pt[0]
    v2 = -(keypoints[1].pt[1] - keypoints[0].pt[1])

    print(f"(u1, v1): \t{u1}\t{v1}")
    print(f"(u2, v2): \t{u2}\t{v2}")

    roll = math.atan2((u1 - u2), (v1 - v2))
    print(f"roll(radian): \t{roll}")
    print(f"roll (degree): \t{roll * 180 / np.pi}")

    tmpu = (u1 - u2) ** 2
    tmpv = (v1 - v2) ** 2
    tmph = (prm.H1 - prm.H2) ** 2

    m = math.sqrt((tmpu + tmpv) / tmph)
    n = (math.sin(roll) * v1 - math.cos(roll) * u1) / prm.D
    k = (u1 * math.sin(roll) + v1 * math.cos(roll) - prm.H1 * m) / prm.D

    print(f"m: \t{m}")
    print(f"n: \t{n}")
    print(f"k: \t{k}")

    if abs(n) > 0.00001:
        tmp1 = m * m + n * n + k * k
        tmp2 = m * m * n * n
        tmp3 = math.sqrt(tmp1 * tmp1 - 4 * tmp2)

        print(f"tmp1: \t{tmp1}")
        print(f"tmp2: \t{tmp2}")
        print(f"tmp3: \t{tmp3}")

        scale = math.sqrt((tmp1 - tmp3) / (2 * tmp2))
    else:
        scale = 1 / math.sqrt(m * m + k * k)

    yaw = math.asin(n * scale)
    pitch = math.asin(k * scale / math.cos(yaw))

    print(f"scale: \t{scale}")
    print(f"pitch: \t{pitch * 180 / np.pi}")
    print(f"yaw: \t{yaw * 180 / np.pi}")

    pyr[0] = pitch * 180 / np.pi
    pyr[1] = yaw * 180 / np.pi
    pyr[2] = roll * 180 / np.pi

def main():
    D = 53.0
    H1 = 125.0 / 2
    H2 = -125.0 / 2
    prm = IPPrm(D, H1, H2)

    src = cv2.imread("1.png", cv2.IMREAD_GRAYSCALE)
    if src is None:
        print("Could not open or find the image")
        return -1

    startX = 1040
    startY = 662
    roi = src[startY:startY + 1802, startX:startX + 800]
    cv2.imwrite("img_roi.png", roi)

    with open("roi_intensity.txt", "w") as outf:
        max_intensity = 0
        for j in range(roi.shape[0]):
            max_int_row = 0
            for i in range(roi.shape[1]):
                intensity = roi[j, i]
                if intensity > max_int_row:
                    max_int_row = intensity
            outf.write(f"{max_int_row}\n")
            if max_int_row > max_intensity:
                max_intensity = max_int_row

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.maxArea = 10000
    params.minArea = 30
    params.filterByCircularity = True
    params.maxCircularity = 1
    params.minCircularity = 0.5
    params.filterByConvexity = True
    params.minConvexity = 0.7
    params.maxConvexity = 1
    params.filterByColor = False
    params.filterByInertia = True
    params.minInertiaRatio = 0.8
    params.minThreshold = 150
    params.maxThreshold = 255

    blob_detector = cv2.SimpleBlobDetector_create(params)
    keypoints = blob_detector.detect(roi)

    img_keypoints = cv2.drawKeypoints(roi, keypoints, None)
    cv2.imshow("Blob keypoints", img_keypoints)

    cv2.imwrite("Blob_keypoint.png", img_keypoints)

    print(f"Number of keypoints detected: \t{len(keypoints)}")
    for kp in keypoints:
        print(f"{kp.pt}\t{kp.size}")

    uv = np.zeros((3, 2))
    for i in range(3):
        uv[i][0] = keypoints[i].pt[0]
        uv[i][1] = keypoints[i].pt[1]
        print(f"{uv[i][0]}\t{uv[i][1]}")

    pyr = [0.0, 0.0, 0.0]
    api_ipb_3uv2pyr(keypoints, pyr, prm)
    print(f"{pyr[0]}\t{pyr[1]}\t{pyr[2]}")

    cv2.waitKey()

if __name__ == "__main__":
    main()
