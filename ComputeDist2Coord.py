import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def linear_regression(a:np.array, b:np.array):
    model = LinearRegression()
    a = a.reshape(-1, 1)
    model.fit(a, b)
    return model

def get_dist2Coord_transform(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    distance = np.array(data["Parallax"]["Distance"])
    x = np.array(data["Parallax"]["X"])
    y = np.array(data["Parallax"]["Y"])
    
    mask = (distance >= 2.6) & (distance <= 15)
    distance = distance[mask]
    x = x[mask]
    y = y[mask]
    
    x = x ** 47
    y = y ** 12
    
    dist2X = linear_regression(distance, x)
    dist2Y = linear_regression(distance, y)
    
    slopeX, interceptX = dist2X.coef_[0], dist2X.intercept_
    slopeY, interceptY = dist2Y.coef_[0], dist2Y.intercept_
    print(f"Slope x: {slopeX}, intercept x: {interceptX}")
    print(f"Slope y: {slopeY}, intercept y: {interceptY}")
    
    plt.figure(figsize=(14, 6))

    # Plot for Distance vs X
    plt.subplot(1, 2, 1)
    plt.plot(distance, x, color='blue', label='X')
    plt.plot(distance, dist2X.predict(distance.reshape(-1, 1)), color='red', label='Fitted line')
    plt.xlabel('Distance')
    plt.ylabel('X')
    plt.title('Distance vs X')
    plt.legend()

    # Plot for Distance vs Y
    plt.subplot(1, 2, 2)
    plt.plot(distance, y, color='green', label='Y')
    plt.plot(distance, dist2Y.predict(distance.reshape(-1, 1)), color='red', label='Fitted line')
    plt.xlabel('Distance')
    plt.ylabel('Y')
    plt.title('Distance vs Y')
    plt.legend()

    plt.show()
    
if __name__ == '__main__':
    json_path = 'cam_calibration_50506.json'
    get_dist2Coord_transform(json_path)