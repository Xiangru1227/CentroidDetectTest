from utils import *

def main():
    D = 53.0
    H1 = 125.0 / 2
    H2 = -125.0 / 2
    prm = Prm(D, H1, H2)
    
    # Modify these parameters to input different datasets
    root_path = '6.26/2.6m'
    sequence = '2'
    distance = 5
    
    smr_coord_normed = [(3.9173226613537447e-115 * distance - 1.1549813265687507e-114) ** (1 / 47), 
                        (7.315308269163082e-24 * distance - 2.2948441817113124e-23) ** (1 / 12)]
    
    print(smr_coord_normed)
    
    _draw = True
    hsv_min = (50, 100, 50)
    hsv_max = (100, 255, 255)
    
    # visualize_hsv_range(hsv_min, hsv_max)
    
    bgr_img = YUV2BGR(root_path, sequence)
    # cv2.imwrite("bgr_img_5m.png", bgr_img)
    # show_image(bgr_img, "BGR image", size=[800,600], position=[500,100])
    
    centroids, area, img_contoured = color_detection(bgr_img, hsv_min, hsv_max, _draw)
    # cv2.imwrite("contoured image_5m.png", img_contoured)
    
    print(f"Number of keypoints detected: {len(centroids)}")
    if len(centroids) == 3:
        print("Coordinates and areas (P1, P3, P2):")
        for i in range(len(centroids)):
            print(f"{centroids[i]}\t{area[i]}")
        print()
    elif len(centroids) > 3:
        print("Image is noisy, adjust filtering thresholds.")
    else:
        print("Detected probe is incomplete, move closer or adjust pose.")
    
    # pyrs = iPb_uv2pyr(centroids, prm)
    # print("Pitch, Yaw, Roll and Scale:")
    # print(pyrs)
    
    # if _draw:
    #     show_image(img_contoured, "Image with contour", size=[800,600], position=[500,200])

if __name__ == "__main__":
    main()
