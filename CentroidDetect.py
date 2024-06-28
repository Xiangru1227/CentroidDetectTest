from utils import *

def main():
    D = 53.0
    H1 = 125.0 / 2
    H2 = -125.0 / 2
    prm = Prm(D, H1, H2)
    
    # Modify these parameters to input different datasets
    root_path = '6.26/15m'
    sequence = '1'
    # distance = 10.227828
    # cam_cal = 'cam_calibration_50506.json'
    
    _draw = False
    hsv_min = (50, 100, 50)
    hsv_max = (100, 255, 255)
    # visualize_hsv_range(hsv_min, hsv_max)
    
    bgr_img = YUV2BGR(root_path, sequence)
    # cv2.imwrite("bgr_img_5m.png", bgr_img)
    # show_image(bgr_img, "BGR image", size=[800,600], position=[500,200])
    
    # coord_normed = find_dist2XY(distance, cam_cal)
    # SMR_coord = [(coord_normed[0] + 1) * bgr_img.shape[1] / 2, (coord_normed[1] + 1) * bgr_img.shape[0] / 2]
    # print(SMR_coord)
    
    SMR_img = bgr_img.copy()
    SMR_coord, img_red = find_red_areas(SMR_img, True)
    # show_image(img_red, "Image with red area contoured", size=[800,600], position=[500,200])
    
    bgr_img = filter_red(bgr_img)
    centroids, area, img_contoured = color_detection(bgr_img, hsv_min, hsv_max, SMR_coord, draw=_draw)
    # cv2.imwrite("contoured image_5m.png", img_contoured)
    
    print(f"Number of keypoints detected: {len(centroids)}")
    print("Coordinates and areas (P1, P3, P2):")
    for i in range(len(centroids)):
        print(f"{centroids[i]}\t{area[i]}")
    
    pyrs = iPb_uv2pyr(centroids, prm)
    print("Pitch, Yaw, Roll and Scale:")
    print(pyrs)
    
    # cv2.circle(img_contoured, (int(SMR_coord[0]), int(SMR_coord[1])), radius=20, color=(255, 0, 0), thickness=10)
    
    print("Crop length:")
    print(np.linalg.norm(SMR_coord - centroids[0]))
    
    if _draw:
        show_image(img_contoured, "Image with contour", size=[800,600], position=[500,200])

if __name__ == "__main__":
    main()
