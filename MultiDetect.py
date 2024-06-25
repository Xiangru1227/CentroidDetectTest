import cv2
import os

from utils import *

def main():
    D = 53.0
    H1 = 125.0 / 2
    H2 = -125.0 / 2
    prm = Prm(D, H1, H2)
    _draw = True
    
    # only change this variable when measuring from different distance!
    distance = 5
    
    list_of_centroidX = [[] for _ in range(3)]
    list_of_centroidY = [[] for _ in range(3)]
    list_of_area = [[] for _ in range(3)]
    list_of_pyrs = [[] for _ in range(4)]
    
    img_output_path = f'6.24/{distance}m/img_output'
    os.makedirs(img_output_path, exist_ok=True)
    cnt_output_path = f'6.24/{distance}m/cnt_output.txt'
    
    for i in range(1, 11):
        imgY_path = os.path.join(f'6.24/{distance}m', str(i), 'src2Y.png')
        imgU_path = os.path.join(f'6.24/{distance}m', str(i), 'src2U.png')
        imgV_path = os.path.join(f'6.24/{distance}m', str(i), 'src2V.png')
        
        imgY = cv2.imread(imgY_path, cv2.IMREAD_GRAYSCALE)
        imgU = cv2.imread(imgU_path, cv2.IMREAD_GRAYSCALE)
        imgV = cv2.imread(imgV_path, cv2.IMREAD_GRAYSCALE)
    
        if imgY is None or imgU is None or imgV is None:
            print(f"Failed to read images from folder {i}")
            continue
    
        height, width = imgY.shape
        imgU = cv2.resize(imgU, (width, height), interpolation=cv2.INTER_LINEAR)
        imgV = cv2.resize(imgV, (width, height), interpolation=cv2.INTER_LINEAR)

        yuv_img = cv2.merge([imgY, imgU, imgV])
        bgr_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)
    
        centroids, img_contoured, keypoints = colorDetection(bgr_img, hsv_min=(35, 50, 80), hsv_max=(105, 255, 255), draw=_draw)
        img_output = os.path.join(img_output_path, f'{i}.png')
        # cv2.imwrite(img_output, img_contoured)
        
        # with open(cnt_output_path, 'a') as f:
        #     f.write(f"Image #{i}\n")
        #     f.write(f"Number of color blocks detected: {len(centroids)}\n")
        #     f.write("Coordinates and areas of centroids (sorted by area from low to high):\n")
        #     for i in range(len(centroids)):
        #         f.write(f"{centroids[i][0]}\t{centroids[i][1]}\n")
        #     f.write('\n')
        
        for j in range(3):
            list_of_centroidX[j].append(centroids[j][0][0])
            list_of_centroidY[j].append(centroids[j][0][1])
            list_of_area[j].append(centroids[j][1])
        
        pyrs = iPb_uv2pyr(keypoints, prm)
        list_of_pyrs[0].append(pyrs[0])
        list_of_pyrs[1].append(pyrs[1])
        list_of_pyrs[2].append(pyrs[2])
        list_of_pyrs[3].append(pyrs[3])
    
    # print(list_of_centroidX)
    # print(list_of_centroidY)
    # print(list_of_area)
    
    print("Standard deviation of P1 coordinate and area:")
    print(f"({np.std(list_of_centroidX[0])}, {np.std(list_of_centroidY[0])}), {np.std(list_of_area[0])}")
    print("Standard deviation of P2 coordinate and area:")
    print(f"({np.std(list_of_centroidX[2])}, {np.std(list_of_centroidY[2])}), {np.std(list_of_area[2])}")
    print("Standard deviation of P3 coordinate and area:")
    print(f"({np.std(list_of_centroidX[1])}, {np.std(list_of_centroidY[1])}), {np.std(list_of_area[1])}")
    print()
    
    print(f"Standard deviation of pitch: {np.std(list_of_pyrs[0])}")
    print(f"Standard deviation of yaw  : {np.std(list_of_pyrs[1])}")
    print(f"Standard deviation of roll : {np.std(list_of_pyrs[2])}")
    print(f"Standard deviation of scale: {np.std(list_of_pyrs[3])}")
    
        
if __name__ == "__main__":
    main()
