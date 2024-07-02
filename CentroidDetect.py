from utils import *

def main():
    D = 53.0
    H1 = 125.0 / 2
    H2 = -125.0 / 2
    prm = Prm(D, H1, H2)
    
    # Modify these parameters to input different datasets
    root_path = '7.1/12.5m'
    sequence = '1'
    distance = 12484.315
    cam_cal = 'cam_calibration_50503.json'
    map = 'distance_mapping.json'
    
    _draw = True
    hsv_min = (50, 50, 50)
    hsv_max = (100, 255, 255)
    # visualize_hsv_range(hsv_min, hsv_max)
    
    bgr_img = YUV2BGR(root_path, sequence)
    # cv2.imwrite("bgr_img_5m.png", bgr_img)
    show_image(bgr_img, "BGR image", size=[800,600], position=[500,200])
    
    SMR_img = bgr_img.copy()
    SMR_coord = find_red_areas(SMR_img, False)
    bgr_img = filter_red(bgr_img)
    ROI_size, LED_filter_size = dist2ROI_size(distance, map)
    ROI_img = crop_image(bgr_img, SMR_coord, ROI_size)
    # show_image(ROI, "ROI", size=[800,600], position=[500,200])
    
    ROI_centroids, ROI_area, ROI_contoured = color_detection(ROI_img, hsv_min, hsv_max, SMR_coord, LED_filter_size * 0.75, draw=_draw)
    for i in range(len(ROI_centroids)):
        ROI_centroids[i] += np.array([SMR_coord[0] - ROI_size, SMR_coord[1] - ROI_size])
    
    # convert normalized coordinate into actual coordinate in the image
    # coord_normed = dist2SMR_coord(distance / 1000, cam_cal)
    # SMR_coord = [(coord_normed[0] + 1) * bgr_img.shape[1] / 2, (coord_normed[1] + 1) * bgr_img.shape[0] / 2]
    # show_image(img_red, "Image with red area contoured", size=[800,600], position=[500,200])
    
    # centroids, area, img_contoured = color_detection(bgr_img, hsv_min, hsv_max, SMR_coord, 50, draw=_draw)
    # cv2.imwrite("contoured image_5m.png", img_contoured)
    
    print(f"Number of keypoints detected: {len(ROI_centroids)}")
    print("Coordinates and areas (P1, P3, P2):")
    for i in range(len(ROI_centroids)):
        print(f"{ROI_centroids[i]}\t{ROI_area[i]}")
        
    # print(f"Number of keypoints detected: {len(centroids)}")
    # print("Coordinates and areas (P1, P3, P2):")
    # for i in range(len(centroids)):
    #     print(f"{centroids[i]}\t{area[i]}")
    
    # pyrs = iPb_uv2pyr(ROI_centroids, prm)
    # print("Pitch, Yaw, Roll and Scale:")
    # print(pyrs)
    
    # mark the expected position of the SMR in the image
    # cv2.circle(ROI_contoured, (int(SMR_coord[0]), int(SMR_coord[1])), radius=5, color=(255, 0, 0), thickness=10)
    
    # print("Crop length:")
    # print(np.linalg.norm(SMR_coord - centroids[0]) + 1.5 * np.sqrt(area[0] / np.pi))
    # print("Min size:")
    # print(np.min(area))
    
    if _draw:
        # show_image(img_contoured, "Image with contour", size=[800,600], position=[500,200])
        show_image(ROI_contoured, "ROI with contour", size=[800,600], position=[500,200])

if __name__ == "__main__":
    main()
