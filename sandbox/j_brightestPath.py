
import brightest_path_lib
import brightest_path_lib.algorithm
import matplotlib.pyplot as plt
import numpy as np
from shapely import LineString

from mapmanagercore import MapAnnotations, MultiImageLoader
from mapmanagercore.benchmark import timer
from mapmanagercore.logger import logger
import mapmanagercore
import mapmanagercore.data


def compareImageDimensions():

    path = 'C:\\Users\\johns\\Documents\\GitHub\\MapManagerCore-Data\\data\\single_timepoint.mmap'
    map2 = MapAnnotations.load(path)

    points = map2.analysisParams.getValue("backgroundRoiGridPoints")
    overlap = map2.analysisParams.getValue("backgroundRoiGridOverlap")
    zSpread = map2.analysisParams.getValue("zSpread")
    channel = map2.analysisParams.getValue('channel')

    # slices = map2.getPixels(time=0, channel=channel, zRange=(42, 48)).data(flattened=False)
    slices = map2.getPixels(time=0, channel=0, zRange=(18, 36))
    logger.info(f"slices {slices}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    logger.info(f"zSpread is {zSpread}")
    z = 45
    timepoint = 0
    _singleTimePoint = map2.getTimePoint(timepoint)

    image = _singleTimePoint.getPixels(channel=channel, z=z, zSpread=0).data(flattened=False) # returning in ndarray form    

    print("image 1:")
    print(image.shape)  # Outputs: (height, width)
    print(image.ndim)   # Outputs: 2

    axes[0].imshow(image)  
    axes[0].set_title("1st Image")
    axes[0].axis("off")  # Hide axes

    image2 = _singleTimePoint.getPixels(channel=channel, z=z, zSpread=10).data(flattened=False) # returning in ndarray form    

    print("image 2:")
    print(image2.shape)  # Outputs: (height, width)
    print(image2.ndim)   # Outputs: 2

    axes[1].imshow(image2)  
    axes[1].set_title("2nd Image")
    axes[1].axis("off")  # Hide

    plt.show()

def testBrightestPathAPI():
    from brightest_path_lib.algorithm import AStarSearch
    import numpy as np
    from skimage import data
    import matplotlib.pyplot as plt

    image = data.cells3d()[30, 0]
    plt.imshow(image, cmap='gray')

    start_point = np.array([10,192]) # [y, x]
    end_point = np.array([198,9])

    # let's show the start and end points
    # plt.imshow(image, cmap='gray')
    # plt.plot(start_point[1], start_point[0], 'og')
    # plt.plot(end_point[1], end_point[0], 'or')

    # plt.show()
    logger.info(f"image is {image}")
    search_algorithm = AStarSearch(image, start_point=start_point, goal_point=end_point)
    brightest_path =search_algorithm.search()

    plt.imshow(image, cmap='gray')
    plt.plot(start_point[1], start_point[0], 'og')
    plt.plot(end_point[1], end_point[0], 'or')
    plt.plot([point[1] for point in search_algorithm.result], [point[0] for point in brightest_path], '-y')
    plt.show()



def brightestPath():


    path = 'C:\\Users\\johns\\Documents\\GitHub\\MapManagerCore-Data\\data\\single_timepoint.mmap'
    # path = '\\Users\\johns\\Documents\\GitHub\\MapManagerCore\\data\\rr30a_s0u.mmap'
    # path = 'C:\\Users\\johns\\Documents\\PyMapManager-Data\\PyMapManager-Data\\one-timepoint\\rr30a_s0_ch1.tif'
    map2 = MapAnnotations.load(path)

    points = map2.analysisParams.getValue("backgroundRoiGridPoints")
    overlap = map2.analysisParams.getValue("backgroundRoiGridOverlap")
    zSpread = map2.analysisParams.getValue("zSpread")
    channel = map2.analysisParams.getValue('channel')

    timepoint = 0
    _singleTimePoint = map2.getTimePoint(timepoint)
    
    segmentId = 6


    # x = 811 # 405
    # y = 939 # 708
    # z = 32 #45

    # 25 (237,614) (237, 631)

    x = 237#
    y = 614
    z = 25 

    speculate: bool = False
    start_point = np.array([x,y,z])
    _singleTimePoint.appendSegmentPoint(segmentId, x, y, z)
    # map2.segments.appendSegmentPoint(segmentId, x, y, z)

    # end_point = np.array([725,424])
    # x = 729 #385
    # y = 908 #693
    # z = 32 #45
    x = 584
    y = 401
    z = 20

    x = 237
    y = 631
    z = 25

    end_point = np.array([x,y,z])
    _singleTimePoint.appendSegmentPoint(segmentId, x, y, z)

    print(_singleTimePoint._segments[:])

    _singleTimePoint._segments[-1]
    fig, ax = plt.subplots(figsize=(10, 10))

    _singleTimePoint = map2.getTimePoint(timepoint)
    map2.segments[[segmentId], "segment"].plot(ax=ax, color="red")

    slices = map2.getPixels(time=0, channel=channel, zRange=(z-3, z+3))
    slices.plot(ax=ax, vmin=300, vmax=1500, alpha=0.3, cmap='CMRmap')
    ax.set_xlim(170, 300)#ax.set_xlim(700, 850) # ax.set_xlim(322, 612)
    ax.set_ylim(100, 200)#ax.set_ylim(840, 1000) # ax.set_ylim(610, 866)
    plt.gca().invert_yaxis()

    plt.plot(start_point[0], start_point[1], 'og')
    plt.plot(end_point[0], end_point[1], 'or')
    plt.show()


def brightestPathIsolated():
    """ Testing logic for astar search before moving into backend functions
    """
    path = 'C:\\Users\\johns\\Documents\\GitHub\\MapManagerCore-Data\\data\\single_timepoint.mmap'
    map2 = MapAnnotations.load(path)

    points = map2.analysisParams.getValue("backgroundRoiGridPoints")
    overlap = map2.analysisParams.getValue("backgroundRoiGridOverlap")
    zSpread = map2.analysisParams.getValue("zSpread")
    channel = map2.analysisParams.getValue('channel')

    # slices = map2.getPixels(time=0, channel=channel, zRange=(42, 48)).data(flattened=False)
    slices = map2.getPixels(time=0, channel=0, zRange=(18, 36))
    logger.info(f"slices {slices}")

    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    logger.info(f"zSpread is {zSpread}")
    z = 45
    timepoint = 0
    _singleTimePoint = map2.getTimePoint(timepoint)
    # image = map2.getPixels(time=0, channel=channel, zRange=(z-3, z+3))
    # plt.imshow(image, cmap='gray')
    # segments = map2.segments
    logger.info(f"z: {z} zSpread: {zSpread} channel {channel}")
    image = _singleTimePoint.getPixels(channel=channel, z=z, zSpread=zSpread).data(flattened=False) # returning in ndarray form    
    start_point = np.array([693,385]) # [y, x]
    end_point = np.array([708,405]) # original
    # end_point = np.array([725,424])

    # print("image 2:")
    # print(image2.shape)  # Outputs: (height, width)
    # print(image2.ndim)   # Outputs: 2

    # below is good
    logger.info(f"image is {image}")
    search_algorithm = brightest_path_lib.algorithm.AStarSearch(image, start_point=start_point, goal_point=end_point)
    brightest_path =search_algorithm.search()
    logger.info(f"brightest_path {brightest_path}")

    # plt.imshow(image, cmap='gray')
    plt.imshow(image)
    plt.plot(start_point[1], start_point[0], 'og')
    plt.plot(end_point[1], end_point[0], 'or')
    plt.plot([point[1] for point in search_algorithm.result], [point[0] for point in brightest_path], '-y')
    # plt.gca().invert_yaxis()
    plt.show()

@timer
def brightestPathIsolated3D():
    path = 'C:\\Users\\johns\\Documents\\GitHub\\MapManagerCore-Data\\data\\single_timepoint.mmap'
    map2 = MapAnnotations.load(path)

    points = map2.analysisParams.getValue("backgroundRoiGridPoints")
    overlap = map2.analysisParams.getValue("backgroundRoiGridOverlap")
    zSpread = map2.analysisParams.getValue("zSpread")
    channel = map2.analysisParams.getValue('channel')

    # slices = map2.getPixels(time=0, channel=channel, zRange=(42, 48)).data(flattened=False)
    slices = map2.getPixels(time=0, channel=0, zRange=(18, 36))
    logger.info(f"slices {slices}")

    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    logger.info(f"zSpread is {zSpread}")
    z = 45
    timepoint = 0
    _singleTimePoint = map2.getTimePoint(timepoint)
    # image = map2.getPixels(time=0, channel=channel, zRange=(z-3, z+3))
    # plt.imshow(image, cmap='gray')
    # segments = map2.segments
    logger.info(f"z: {z} zSpread: {zSpread} channel {channel}")
    image = _singleTimePoint.getPixels(channel=channel, z=z, zSpread=zSpread, threeD = True).data(flattened=False) # returning in ndarray form    
    
    import math
    imageZ, imageX, imageY = image.shape
    reIndexZ = math.floor(imageZ/2)
    logger.info(f"reIndexZ {reIndexZ}")
    
    # start_point = np.array([reIndexZ,385,693]) # [z,x,y]
    # end_point = np.array([reIndexZ,405,708])

        
    start_point = np.array([reIndexZ,693,385]) # actually [z,y,x], API documentation is wrong!?
    end_point = np.array([reIndexZ,708,405])

    # image = image[3]
    print("image:")
    print(image.shape)  # Outputs: (height, width)
    print(image.ndim)   # Outputs: 3

    # below is good
    logger.info(f"image is {image}")
    search_algorithm = brightest_path_lib.algorithm.AStarSearch(image, start_point=start_point, goal_point=end_point)
    # search_algorithm = brightest_path_lib.algorithm.NBAStarSearch(image, start_point=start_point, goal_point=end_point)
    brightest_path =search_algorithm.search()
    logger.info(f"brightest_path {brightest_path}")

    fig, ax = plt.subplots(figsize=(10, 10))
    slices = map2.getPixels(time=0, channel=channel, zRange=(z-3, z+3))
    slices.plot(ax=ax, vmin=300, vmax=1500, alpha=0.3, cmap='CMRmap')
    ax.set_xlim(322, 612)
    ax.set_ylim(610, 866)

    plt.plot(start_point[2], start_point[1], 'og')
    plt.plot(end_point[2], end_point[1], 'or')
    plt.plot([point[2] for point in search_algorithm.result], [point[1] for point in brightest_path], '-y')
    plt.gca().invert_yaxis()
    plt.show()


def boundingBox():
    path = 'C:\\Users\\johns\\Documents\\GitHub\\MapManagerCore-Data\\data\\single_timepoint.mmap'
    map2 = MapAnnotations.load(path)

    points = map2.analysisParams.getValue("backgroundRoiGridPoints")
    overlap = map2.analysisParams.getValue("backgroundRoiGridOverlap")
    zSpread = map2.analysisParams.getValue("zSpread")
    channel = map2.analysisParams.getValue('channel')

    # slices = map2.getPixels(time=0, channel=channel, zRange=(42, 48)).data(flattened=False)
    slices = map2.getPixels(time=0, channel=0, zRange=(18, 36))
    logger.info(f"slices {slices}")

    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    logger.info(f"zSpread is {zSpread}")
    z = 32
    timepoint = 0
    _singleTimePoint = map2.getTimePoint(timepoint)
    # image = map2.getPixels(time=0, channel=channel, zRange=(z-3, z+3))
    # plt.imshow(image, cmap='gray')
    # segments = map2.segments
    logger.info(f"z: {z} zSpread: {zSpread} channel {channel}")
    image = _singleTimePoint.getPixels(channel=channel, z=z, zSpread=zSpread, threeD = True).data(flattened=False) # returning in ndarray form    
    # cropped_image = image[:, 908:940, 729:812]

    # Bounding box coordinates
    boundingBoxRange = 0
    x_min, x_max = 729 - boundingBoxRange, 811 + boundingBoxRange # X-axis
    y_min, y_max = 908 - boundingBoxRange, 939 + boundingBoxRange # Y-axis
    z_min, z_max = 32, 32    # Z-axis (single slice at Z=32)
    cropped_image = image[3, y_min:y_max+1, x_min:x_max+1]
    print("cropped_image", cropped_image)

    plt.plot(y_max,  x_max , 'og')  
    plt.plot(y_min, x_min, 'or')

    # fig, ax = plt.subplots(figsize=(10, 10))
    # image.plot(ax=ax)
    # plt.figure(figsize=(8, 6))
    plt.imshow(cropped_image)
    plt.show()
    
# image is the same
#

def speedTest():

    path = 'C:\\Users\\johns\\Documents\\GitHub\\MapManagerCore-Data\\data\\single_timepoint.mmap'
    map2 = MapAnnotations.load(path)

    points = map2.analysisParams.getValue("backgroundRoiGridPoints")
    overlap = map2.analysisParams.getValue("backgroundRoiGridOverlap")
    zSpread = map2.analysisParams.getValue("zSpread")
    channel = map2.analysisParams.getValue('channel')

    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    logger.info(f"zSpread is {zSpread}")
    z = 32
    timepoint = 0
    _singleTimePoint = map2.getTimePoint(timepoint)

    boundingBoxRange = 0
    # x1, y1 = 729, 811  # X-axis
    # x2, y2 = 908, 939  # Y-axis
    x1, y1 = 729, 908  # X-axis
    x2, y2 = 811, 939  # Y-axis

    import time
    start = time.time()  # Start time
    roughSegment = LineString([[x1, y1, z], [x2, y2, z]])
    brightestPath = _singleTimePoint.brightestPath(roughSegment, True, z)
    logger.info(f"brightestPath {brightestPath}")

    end = time.time()  # End time
    # print(f"Execution Time: {end - start:.4f} seconds")

if __name__ == '__main__':
    import time

    start = time.time()  # Start time
    # boundingBox()
    # brightestPath()
    # brightestPathIsolated()
    # brightestPathIsolated3D()
    # testBrightestPathAPI()
    speedTest()
    # brightestPath()
    end = time.time()  # End time
    print(f"Execution Time: {end - start:.4f} seconds")

# move out time tests here
# clean up code.
# commit
# merge with suhaybs code
# 811 939 32, 729 908 32)

# LINESTRING Z (811 939 32, 791 932 31, 769 927 32, 748 918 33, 729 908 32)


# Astar
# debug 1 (811 939 32, 791 932 31, 769 927 32, 748 918 33, 729 908 32)
# debug 2(729 908 32, 748 918 33, 769 927 32, 791 932 31, 811 939 32)