import glob
import os
import brightest_path_lib
import brightest_path_lib.algorithm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
import time

from PIL import Image
from shapely import LineString, Point, Polygon
from brightest_path_lib.algorithm import AStarSearch
# from brightest_path_lib.algorithm import BidirectionalAStarSearch # acceleration
# from brightest_path_lib.algorithm import A

from mapmanagercore import MapAnnotations
from mapmanagercore.benchmark import timer
from mapmanagercore.logger import logger
import mapmanagercore
import mapmanagercore.data


def brightestPath():

    pointsPath = "C:\\Users\\johns\\Downloads\\sparse_segment.csv"
    imgPath = "C:\\Users\\johns\\Documents\\PyMapManager-Data\\PyMapManager-Data\\one-timepoint\\rr30a_s0_ch2.tif"

    import tifffile as tiff

    zMin = 28
    zMax = 31
    # img = Image.open(imgPath)
    # img_array = np.array(img)
    
    img_array = tiff.imread(imgPath)

    # print("Image shape:", img_array.shape)
    # img_array = img_array[28:32]  # includes 28, 29, 30, 31
    # image = image[:, y_min:y_max+1, x_min:x_max+1]
    # zMin = 28-2
    # zMax = 31+3
    img_array = img_array[zMin:zMax+1, :, :]
    # print(img_array.shape)  # should be (4, Y, X)

    # Load CSV
    df = pd.read_csv(pointsPath)

    # Extract points as tuples (x, y, z)
    # df[['z', 'x', 'y']] = df[['z', 'x', 'y']].astype(int)
    points = df[['z', 'x', 'y']].values
    points[:, 0] -= 28
    pixelDistanceList = []
    timeList = []
    
    # plt.imshow(img_array, cmap='gray')
    # plt.title("Grayscale Image")
    # plt.axis('off')  # optional: hide axis ticks
    # plt.show()
    # return

    # for i in range(1, len(points)):
    for i in range(1, len(points),3):
        start = points[0]
        end = points[i]
        logger.info(f"start {start} end {end}")
        
        startTime = time.time()  # Start time
        # search_algorithm = AStarSearch(img_array, start_point=start, goal_point=end)
        search_algorithm = AStarSearch(img_array, start_point=end, goal_point=start)
        # search_algorithm = BidirectionalAStarSearch(img_array, start_point=end, goal_point=start)
        brightest_path = search_algorithm.search()
        # logger.info(f"brightest_path {brightest_path}")
        endTime = time.time()  # End time

        executionTime = endTime - startTime
        print(f"Execution Time: {executionTime:.4f} seconds")

        # logger.info(f"test Point(start) {Point(start)}")
        pixelDistance = shapely.distance(Point(start), Point(end))
        # logger.info(f"pixelDistance {pixelDistance}")

        pixelDistanceList.append(pixelDistance)
        timeList.append(executionTime)
        # break

    df_results = pd.DataFrame({
        'PixelDistance': pixelDistanceList,
        'ExecutionTime': timeList
    })
    df_results.to_csv('execution_vs_distance_windows_no_acceleration.csv', index=False)


def plotGraph():
    csv_dir = '../mapmanagercore/'
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))

    # Read and plot each CSV
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        
        # Assuming the CSV has columns like 'distance' and 'execution_time'
        # Update these if your actual column names are different
        x = df['PixelDistance']
        y = df['ExecutionTime']
        
        label = os.path.basename(csv_file).replace('.csv', '')
        plt.plot(x, y, label=label)

    # Customize plot
    plt.xlabel('Distance')
    plt.ylabel('Execution Time')
    plt.title('Execution Time vs Distance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

        
if __name__ == '__main__':
      
    # start = time.time()  # Start time
    # brightestPath()

    # end = time.time()  # End time

    # executionTime = end - start
    # print(f"Total Execution Time: {executionTime:.4f} seconds")

    plotGraph()