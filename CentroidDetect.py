import cv2
from utils import *

def main():
    # initialize iProbe parameters (get from calibration)
    D = 53.0
    H1 = 125.0 / 2
    H2 = -125.0 / 2
    prm = Prm(D, H1, H2)
    hsv_min = (35, 50, 80)
    hsv_max = (105, 255, 255)
    _draw = True
    
    # define paths of yuv420 images
    imgY_path = '6.24/5m/1/src2Y.png'
    imgU_path = '6.24/5m/1/src2U.png'
    imgV_path = '6.24/5m/1/src2V.png'
    
    # read images in grayscale
    imgY = cv2.imread(imgY_path, cv2.IMREAD_GRAYSCALE)
    imgU = cv2.imread(imgU_path, cv2.IMREAD_GRAYSCALE)
    imgV = cv2.imread(imgV_path, cv2.IMREAD_GRAYSCALE)
    
    if imgY is None or imgU is None or imgV is None:
        return -1
    
    height, width = imgY.shape
    imgU = cv2.resize(imgU, (width, height), interpolation=cv2.INTER_LINEAR)
    imgV = cv2.resize(imgV, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # convert yuv image to bgr
    yuv_img = cv2.merge([imgY, imgU, imgV])
    bgr_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)
    # cv2.imwrite("bgr_img_5m.png", bgr_img)
    
    bgr_img = filter_red_color(bgr_img)
    
    cv2.namedWindow("BGR image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("BGR image", 800, 600)
    cv2.imshow("BGR image", bgr_img)
    cv2.waitKey(0)
    
    centroids, img_contoured, keypoints = colorDetection(bgr_img, hsv_min, hsv_max, _draw)
    # cv2.imwrite("contoured image_5m.png", img_contoured)
    
    print(f"Number of keypoints detected: {len(centroids)}\n")
    print("Coordinates and areas of centroids (sorted by area from low to high):")
    for i in range(len(centroids)):
        print(f"{centroids[i][0]}\t{centroids[i][1]}")
    print()
    
    # pyrs = iPb_uv2pyr(keypoints, prm)
    # print("Pitch, Yaw, Roll and Scale:")
    # print(pyrs)
    
    if _draw:
        cv2.namedWindow("Image with contour", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image with contour", 800, 600)
        cv2.imshow("Image with contour", img_contoured)
        cv2.waitKey(0)
    

    
    
    # startX = 0
    # startY = 0
    # roi = img[startY:startY + 1802, startX:startX + 800]

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
