from utils import *

def main():
    D = 53.0
    H1 = 125.0 / 2
    H2 = -125.0 / 2
    prm = Prm(D, H1, H2)
    
    # Modify these parameters to input different datasets
    root_path = '6.24/5m'
    sequence = '1'
    
    _draw = True
    hsv_min = (50, 100, 60)
    hsv_max = (100, 255, 255)
    
    # visualize_hsv_range(hsv_min, hsv_max)
    
    bgr_img = YUV2BGR(root_path, sequence)
    # cv2.imwrite("bgr_img_5m.png", bgr_img)
    # show_image(bgr_img, "BGR image")
    
    centroids, area, img_contoured = color_detection(bgr_img, hsv_min, hsv_max, _draw)
    # cv2.imwrite("contoured image_5m.png", img_contoured)
    
    print(f"Number of keypoints detected: {len(centroids)}\n")
    if len(centroids) == 3:
        print("Coordinates and areas of centroids (sorted by position from high to low):")
        for i in range(len(centroids)):
            print(f"{centroids[i]}\t{area[i]}")
        print()
    else: print("Image is noisy, adjust filtering thresholds.")
    
    pyrs = iPb_uv2pyr(centroids, prm)
    print("Pitch, Yaw, Roll and Scale:")
    print(pyrs)
    
    # if _draw:
    #     show_image(img_contoured, "Image with contour")

if __name__ == "__main__":
    main()
