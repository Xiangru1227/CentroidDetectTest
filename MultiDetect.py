from utils import *

def main():
    D = 53.0
    H1 = 125.0 / 2
    H2 = -125.0 / 2
    prm = Prm(D, H1, H2)
    
    _draw = True
    hsv_min = (50, 100, 60)
    hsv_max = (100, 255, 255)
    
    # Modify this to input different datasets
    root_path = '6.24/5m'
    
    list_of_centroids = [[] for _ in range(3)]
    list_of_area = [[] for _ in range(3)]
    list_of_pyrs = [[] for _ in range(4)]
    
    img_output_path = os.path.join(root_path, 'img_output')
    os.makedirs(img_output_path, exist_ok=True)
    cnt_output_path = os.path.join(root_path, 'cnt_output.txt')
    
    for i in range(1, 11):
        bgr_img = YUV2BGR(root_path, str(i))
    
        centroids, areas, img_contoured = color_detection(bgr_img, hsv_min, hsv_max, draw=_draw)
        img_output = os.path.join(img_output_path, f'{i}.png')
        
        # cv2.imwrite(img_output, img_contoured)
        # with open(cnt_output_path, 'a') as f:
        #     f.write(f"Image #{i}\n")
        #     f.write(f"Number of color blocks detected: {len(centroids)}\n")
        #     f.write("Coordinates and areas of centroids (sorted by position from high to low):\n")
        #     for i in range(len(centroids)):
        #         f.write(f"{centroids[i]}\t{areas[i]}\n")
        #     f.write('\n')
        
        for j in range(3):
            list_of_centroids[j].append(centroids[j])
            list_of_area[j].append(areas[j])
        
        pyrs = iPb_uv2pyr(centroids, prm)
        list_of_pyrs[0].append(pyrs[0])
        list_of_pyrs[1].append(pyrs[1])
        list_of_pyrs[2].append(pyrs[2])
        list_of_pyrs[3].append(pyrs[3])
    
    std_dev = []
    for row in list_of_centroids:
        std_dev.append((np.std([point[0] for point in row]), np.std([point[1] for point in row])))
    
    print("Standard deviation of coordinates and areas (P1, P3, P2):")
    for i in range(3):
        print(f"{std_dev[i]}\t{np.std(list_of_area[i])}")
    print()
    
    print("Standard deviation of pitch, yaw, roll and scale:")
    for i in range(len(pyrs)):
        print(np.std(list_of_pyrs[i]))
    
if __name__ == "__main__":
    main()
