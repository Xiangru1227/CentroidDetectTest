import cv2
import math
import imutils
import numpy as np

class Prm:
    # stored iProbe parameters
    def __init__(self, D, H1, H2):
        self.D = D
        self.H1 = H1
        self.H2 = H2

def api_ipb_3uv2pyr(keypoints, pyr, prm:Prm):
    # compute (u1,v1) and (u2,v2), coordinates of P1 and P2 in P3 frame
    u1 = keypoints[2].pt[0] - keypoints[0].pt[0]
    v1 = -(keypoints[2].pt[1] - keypoints[0].pt[1])
    u2 = keypoints[1].pt[0] - keypoints[0].pt[0]
    v2 = -(keypoints[1].pt[1] - keypoints[0].pt[1])
    
    print(f"(u1, v1): ({u1}, \t{v1})")
    print(f"(u2, v2): ({u2}, \t{v2})")

    # compute roll for general cases (in degree)
    roll = (math.atan2((u1 - u2), (v1 - v2))) * 180 / np.pi

    # cos(pitch) = m * scale
    m = math.sqrt(((u1 - u2) ** 2 + (v1 - v2) ** 2) / (prm.H1 - prm.H2) ** 2)
    # sin(yaw) = n * scale
    n = (math.sin(roll) * v1 - math.cos(roll) * u1) / prm.D
    #  sin(pitch) * cos(yaw) = k * scale
    k = (u1 * math.sin(roll) + v1 * math.cos(roll) - prm.H1 * m) / prm.D
    
    print(f"m: {m}")
    print(f"n: {n}")
    print(f"k: {k}")

    # compute scale
    if abs(n) > 0.00001:
        scale = math.sqrt(
            (m ** 2 + n ** 2 + k ** 2 - math.sqrt(m ** 2 + n ** 2 + k ** 2 ** 2 - 4 * (m ** 2) * (n ** 2))) / 
            (2 * (m ** 2) * (n ** 2)))
    
    # when Probe is vertical
    else:
        scale = 1 / math.sqrt(m ** 2 + k ** 2)

    # compute pitch and yaw
    pitch = (math.asin(k * scale / math.cos(yaw))) * 180 / np.pi
    yaw = (math.asin(n * scale)) * 180 / np.pi

    print(f"scale = {scale}")
    print(f"roll = {roll}")
    print(f"pitch = {pitch}")
    print(f"yaw =: {yaw}")

    pyr[0] = pitch
    pyr[1] = yaw
    pyr[2] = roll
    
def drawContour(img, hsv_min, hsv_max, draw=False):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, hsv_min, hsv_max)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))

    # Find contours in the mask
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    centroids = []
    for c in contours:
        # Compute the center of the contour
        M = cv2.moments(c)
        if M["m00"] == 0:
            cX = M["m10"]
            cY = M["m01"]
        else:
            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]
        centroids.append(np.array([cX, cY]))

    if draw:
        cv2.drawContours(img, contours, -1, (0, 0, 255), thickness=2)
        return np.array(centroids), img
    
    else:
        return np.array(centroids), None

def main():
    # initialize iProbe parameters (get from calibration)
    D = 53.0
    H1 = 125.0 / 2
    H2 = -125.0 / 2
    prm = Prm(D, H1, H2)
    
    # define paths of yuv420 images
    imgY_path = '10m/src2Y.png'
    imgU_path = '10m/src2U.png'
    imgV_path = '10m/src2V.png'
    
    # read images in grayscale
    imgY = cv2.imread(imgY_path, cv2.IMREAD_GRAYSCALE)
    imgU = cv2.imread(imgU_path, cv2.IMREAD_GRAYSCALE)
    imgV = cv2.imread(imgV_path, cv2.IMREAD_GRAYSCALE)
    
    if imgY is None or imgU is None or imgV is None:
        return -1
    
    # cv2.namedWindow("imageY", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("imageU", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("imageV", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("imageY", 800, 600)
    # cv2.resizeWindow("imageU", 800, 600)
    # cv2.resizeWindow("imageV", 800, 600)
    # cv2.imshow("imageY", imgY)
    # cv2.imshow("imageU", imgU)
    # cv2.imshow("imageV", imgV)
    # cv2.waitKey(0)
    
    height, width = imgY.shape
    imgU = cv2.resize(imgU, (width, height), interpolation=cv2.INTER_LINEAR)
    imgV = cv2.resize(imgV, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # convert yuv image to bgr
    yuv_img = cv2.merge([imgY, imgU, imgV])
    bgr_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)
    
    # cv2.namedWindow("BGR image", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("BGR image", 800, 600)
    # cv2.imshow("BGR image", bgr_img)
    # cv2.waitKey(0)
    
    centroids, result_img = drawContour(bgr_img, hsv_min=(35, 120, 80), hsv_max=(105, 255, 255), draw=True)
    
    cv2.namedWindow("Image with contour", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image with contour", 800, 600)
    cv2.imshow("Image with contour", result_img)
    cv2.waitKey(0)

    
    
    # startX = 0
    # startY = 0
    # roi = img[startY:startY + 1802, startX:startX + 800]
    # # cv2.imwrite("img_roi.png", roi)

    # with open("roi_intensity.txt", "w") as outf:
    #     max_intensity = 0
    #     for j in range(roi.shape[0]):
    #         max_int_row = 0
    #         for i in range(roi.shape[1]):
    #             intensity = roi[j, i]
    #             if intensity > max_int_row:
    #                 max_int_row = intensity
    #         outf.write(f"{max_int_row}\n")
    #         if max_int_row > max_intensity:
    #             max_intensity = max_int_row

    # params = cv2.SimpleBlobDetector_Params()
    # params.filterByArea = True
    # params.maxArea = 10000
    # params.minArea = 30
    # params.filterByCircularity = True
    # params.maxCircularity = 1
    # params.minCircularity = 0.5
    # params.filterByConvexity = True
    # params.minConvexity = 0.7
    # params.maxConvexity = 1
    # params.filterByColor = False
    # params.filterByInertia = True
    # params.minInertiaRatio = 0.8
    # params.minThreshold = 150
    # params.maxThreshold = 255

    # blob_detector = cv2.SimpleBlobDetector_create(params)
    # keypoints = blob_detector.detect(roi)

    # img_keypoints = cv2.drawKeypoints(roi, keypoints, None)
    # cv2.imshow("Blob keypoints", img_keypoints)
    # cv2.waitKey(1)
    # # cv2.imwrite("Blob_keypoint.png", img_keypoints)

    # print(f"Number of keypoints detected: \t{len(keypoints)}")
    # for kp in keypoints:
    #     print(f"{kp.pt}\t{kp.size}")

    # uv = np.zeros((len(keypoints), 2))
    # for i in range(len(keypoints)):
    #     uv[i][0] = keypoints[i].pt[0]
    #     uv[i][1] = keypoints[i].pt[1]
    #     print(f"{uv[i][0]}\t{uv[i][1]}")

    # pyr = [0.0, 0.0, 0.0]
    # api_ipb_3uv2pyr(keypoints, pyr, prm)
    # print(f"{pyr[0]}\t{pyr[1]}\t{pyr[2]}")

    # cv2.waitKey()

if __name__ == "__main__":
    main()
