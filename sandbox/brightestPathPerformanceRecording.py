import glob
import os
import brightest_path_lib
import brightest_path_lib.algorithm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
import time
import tifffile as tiff
from PIL import Image
from shapely import LineString, Point, Polygon

# For no acceleration/ transonic
from brightest_path_lib.algorithm import AStarSearch, NBAStarSearch

# For numba acceleration
# from brightest_path_lib.algorithm import BidirectionalAStarSearch 

from mapmanagercore import MapAnnotations
from mapmanagercore.benchmark import timer
from mapmanagercore.logger import logger
import mapmanagercore
import mapmanagercore.data


def brightestPathRecording():
    """ Loads csv and tif image to perform the brightest path tracing on every 3rd point
    Pixel distance (between start and end points) and execution time on each tracing calculated and saved into a csv file
    afterwardss
    """

    pointsPath = "C:\\Users\\johns\\Downloads\\sparse_segment.csv"
    imgPath = "C:\\Users\\johns\\Documents\\PyMapManager-Data\\PyMapManager-Data\\one-timepoint\\rr30a_s0_ch2.tif"

    # Slice Range
    zMin = 28
    zMax = 31

    # Minimizing image passed in
    img_array = tiff.imread(imgPath)
    img_array = img_array[zMin:zMax+1, :, :]

    # Load CSV
    df = pd.read_csv(pointsPath)

    points = df[['z', 'x', 'y']].values # Format points 
    points[:, 0] -= 28 # Reduce z values of points to match image z indexing
    pixelDistanceList = []
    timeList = []
    
    for i in range(0, len(points),3):
        start = points[0]
        end = points[i]
        logger.info(f"start {start} end {end}")
        
        startTime = time.time()  # Individual Tracing Start time

        # Note: for the search Algorithm, I am using "end" as the start_point and "start" as the goal_point
        # Reasoning: The search algorithm performs better and this is the format used in MapManagerCore
        # due to the order of how points are saved within the backend

        # --------- Comment out one or the other ---------
        # For no acceleration/ transonic
        # search_algorithm = AStarSearch(img_array, start_point=end, goal_point=start)
        search_algorithm = NBAStarSearch(img_array, start_point=end, goal_point=start)
        # For numba Acceleration
        # search_algorithm = BidirectionalAStarSearch(img_array, start_point=start, goal_point=end)
        # -------- --------- ---------- --------- --------

        brightest_path = search_algorithm.search()
        # logger.info(f"brightest_path {brightest_path}")

        endTime = time.time()  # Individual Tracing End time
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
    df_results.to_csv('execution_vs_distance_windows_nba_new_acceleration.csv', index=False)

def plotGraph():
    """ Plot csv files that were saved from brightestPathRecording
    """
    # csv_dir = '../mapmanagercore/'
    csv_dir = '../../../Desktop/brightest_pathCSVs'
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))

    # Read and plot each CSV
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        x = df['PixelDistance']
        y = df['ExecutionTime']
        
        label = os.path.basename(csv_file).replace('.csv', '')
        plt.plot(x, y, label=label)

    # Customize plot
    plt.xlabel('Distance (Pixels)')
    plt.ylabel('Execution Time (Seconds)')
    plt.title('Execution Time vs Distance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
      
    start = time.time()  # Entire start time
    brightestPathRecording()
    end = time.time()  # Entire end time

    executionTime = end - start
    print(f"Total Execution Time: {executionTime:.4f} seconds")

    # plotGraph()