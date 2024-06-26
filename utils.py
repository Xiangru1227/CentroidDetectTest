import os
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

def YUV2BGR(root_path, sequence):
    imgY_path = os.path.join(root_path, sequence, 'src2Y.png')
    imgU_path = os.path.join(root_path, sequence, 'src2U.png')
    imgV_path = os.path.join(root_path, sequence, 'src2V.png')
    
    imgY = cv2.imread(imgY_path, cv2.IMREAD_GRAYSCALE)
    imgU = cv2.imread(imgU_path, cv2.IMREAD_GRAYSCALE)
    imgV = cv2.imread(imgV_path, cv2.IMREAD_GRAYSCALE)
    
    if imgY is None or imgU is None or imgV is None:
        return -1
    
    height, width = imgY.shape
    imgU = cv2.resize(imgU, (width, height), interpolation=cv2.INTER_LINEAR)
    imgV = cv2.resize(imgV, (width, height), interpolation=cv2.INTER_LINEAR)
    
    yuv_img = cv2.merge([imgY, imgU, imgV])
    bgr_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)
    
    bgr_img = filter_red(bgr_img)
    
    return bgr_img
        
def color_detection(img, hsv_min, hsv_max, draw=False):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, hsv_min, hsv_max)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))

    # Find contours in the mask
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    
    centroids_and_areas = []
    filtered_contours = []
    
    for c in contours:
        # Compute the center of the contour
        M = cv2.moments(c)
        # filter out noises
        if M["m00"] > 50 and 0.7 <= circularity(c) <= 1.0:
            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]
            
            filtered_contours.append(c)
            area = cv2.contourArea(c)
            centroids_and_areas.append((np.array([cX, cY]), area))
            
    centroids_and_areas.sort(key=lambda x: x[0][1])
    centroids = [i[0] for i in centroids_and_areas]
    areas = [i[1] for i in centroids_and_areas]

    if draw:
        cv2.drawContours(img, filtered_contours, -1, (0, 0, 255), thickness=2)
        keypoints = [cv2.KeyPoint(x=pt[0], y=pt[1], size=10) for pt in centroids]
        img = cv2.drawKeypoints(img, keypoints, None, color=(0, 0, 255))
        return centroids, areas, img
    else:
        return centroids, areas, None
    
def filter_red(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    mask = mask1 + mask2

    filtered_img = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
    return filtered_img
    
def circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    return circularity
    
def iPb_uv2pyr(keypoints, prm:Prm):
    pyrs = [0.0, 0.0, 0.0, 0.0]
    
    # compute (u1,v1) and (u2,v2), coordinates of P1 and P2 in P3 frame
    u1 = keypoints[0][0] - keypoints[1][0]
    v1 = -(keypoints[0][1] - keypoints[1][1])
    u2 = keypoints[2][0] - keypoints[1][0]
    v2 = -(keypoints[2][1] - keypoints[1][1])

    # compute roll for general cases (in degree)
    roll = (math.atan2((u1 - u2), (v1 - v2))) * 180 / np.pi
    sr = math.sin(roll * np.pi / 180)
    cr = math.cos(roll * np.pi / 180)

    m = math.sqrt(((u1 - u2) ** 2 + (v1 - v2) ** 2) / (prm.H1 - prm.H2) ** 2)
    n = (sr * v1 - cr * u1) / prm.D
    k = (sr * u1 + cr * v1 - prm.H1 * m) / prm.D

    if abs(n) > 0.00001:
        temp1 = m ** 2 + n ** 2 + k ** 2
        temp2 = m ** 2 * n ** 2
        ss = (temp1 - math.sqrt(temp1 ** 2 - 4 * temp2)) / (2 * temp2)
        scale = math.sqrt(ss)
    # when Probe is vertical
    else:
        scale = 1 / math.sqrt(m ** 2 + k ** 2)

    # compute pitch and yaw (in degree)
    yaw = (math.asin(n * scale)) * 180 / np.pi
    pitch = (math.asin(k * scale / math.cos((yaw / 180) * np.pi))) * 180 / np.pi

    pyrs[0] = pitch
    pyrs[1] = yaw
    pyrs[2] = roll
    pyrs[3] = scale
    
    return pyrs

def show_image(img, name, size, position):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, size[0], size[1])
    cv2.moveWindow(name, position[0], position[1])
    cv2.imshow(name, img)
    cv2.waitKey(0)

def visualize_hsv_range(hsv_min, hsv_max):
    gradient = np.zeros((100, 500, 3), dtype=np.uint8)

    for i in range(gradient.shape[1]):
        ratio = i / gradient.shape[1]
        hsv_color = [
            int(hsv_min[j] * (1 - ratio) + hsv_max[j] * ratio)
            for j in range(3)
        ]
        gradient[:, i, :] = hsv_color

    bgr_gradient = cv2.cvtColor(gradient, cv2.COLOR_HSV2BGR)

    cv2.imshow('HSV Visualization', bgr_gradient)
    cv2.waitKey(0)