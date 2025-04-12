import math
from typing import Union

import brightest_path_lib.algorithm
from matplotlib import pyplot as plt
import numpy as np
import scipy
from scipy.interpolate import splprep, splev
from mapmanagercore.utils import injectLine
from shapely.geometry import LineString, Point
import brightest_path_lib
from mapmanagercore.logger import logger

# abb had to add this
from .base import SingleTimePointAnnotationsBase

class AnnotationsSegments(SingleTimePointAnnotationsBase):
    def optimizeSegment(self, roughSegment: LineString, segment: LineString = None, updatedIdx: int = None, live: bool = False, 
                        z: int = None) -> Union[LineString, None]:
        if segment and len(roughSegment.coords) > 2:
            logger.info(f"more than two points")
            if updatedIdx > len(roughSegment.coords) - 1:
                logger.info(f"injecting line")
                if updatedIdx == 0:
                    return LineString([])
                return injectLine(segment, LineString([]), Point(roughSegment.coords[-1]), None)

            left = roughSegment.coords[updatedIdx -
                                       1] if updatedIdx > 0 else None
            point = roughSegment.coords[updatedIdx]
            right = roughSegment.coords[updatedIdx +
                                        1] if updatedIdx < len(roughSegment.coords) - 1 else None

            points = []
            if left:
                logger.info(f"going left")
                leftTracing = self.brightestPath(
                    LineString([left, point]), live, z)
                points = list(leftTracing.coords)
            if right:
                logger.info(f"going right")
                rightTracing = self.brightestPath(
                    LineString([point, right]), live, z)
                points.extend(rightTracing.coords)

            left = Point(left) if left else None
            right = Point(right) if right else None

            segment = injectLine(segment, LineString(
                points), left, right)
        else:
            logger.info(f"here with just 2 points")
            logger.info(f"roughSegment {roughSegment} segment {segment}")
            if roughSegment.coords[0] == roughSegment.coords[1]: # ensure same point is not clicked/ added twiced
                logger.info(f"returning none for optimized segment")
                return None
            
            segment = self.brightestPath(roughSegment, live, z)

        return segment.simplify(0.5)
        # return segment

    def brightestPath(self, roughSegment: LineString, live: bool = False, z: int = None):
        """
        Args:
            roughSegment:  LineString([left, point]) or LineString([point, right])
            live: if live use brightest path tracing to return a set of points from roughSegment[0] to roughSegment[1]
            z: current z Slice
        """
        logger.info(f"roughSegment coming in {roughSegment}")
        # if live:
        #     # TODO: Consider adding the mutation type along with the prior result if we can use it to speed things up
        #     return None

        logger.error('abb turned off')
        return roughSegment
    
        zSpread = self.analysisParams.getValue('zSpread')
        channel = self.analysisParams.getValue('channel')  # 1 based

        # 3D
        image = self.getPixels(channel=channel,
                               zSpread=zSpread,  # abb swapped order
                               z=z,
                               threeD = True).data(flattened=False) # returning in ndarray form

        logger.info(f"channel:{channel} z:{z} zSpread:{zSpread} live:{live} image.shape:{image.shape}")

        x1,y1,z1 = roughSegment.coords[0] # last point added
        x2,y2,z2 = roughSegment.coords[1] # first point added
        # logger.info(f"roughSegment.coords[0] {roughSegment.coords[0]}")
        # logger.info(f"roughSegment.coords[1] {roughSegment.coords[1]}")
        
        # For 3D:
        # get a median index since image does not retain original segment row indexes
        imageZ, imageX, imageY = image.shape # (z, x, y)
        reIndexZ = math.floor(imageZ/2)
        logger.info(f"reIndexZ {reIndexZ}")

        # Testing Restriction of image to a bounding box
        boundingBoxRange = 0
        x_min, x_max = int(min(x1,x2) - boundingBoxRange), int(max(x1,x2) + boundingBoxRange) # X-axis
        y_min, y_max = int(min(y1,y2) - boundingBoxRange), int(max(y1,y2) + boundingBoxRange) # Y-axis

        # Restrict image to a bounding box of the rough Segment
        image = image[:, y_min:y_max+1, x_min:x_max+1]
        slices = image.shape[0]
        logger.info(f"total slices are {slices}")
        height, width = image.shape[1], image.shape[2]  # Get Y (height) and X (width)
        top_left = (0, 0)  # Always starts at (Y=0, X=0)
        newImageSize = (height - 1, width - 1)  # Last Y and X index

        if live:

            # After testing, I found start_point and goal_point order matters for A/NBAStarSearch
            # start_point within AstarSearch should always be the last point added (appended)
            # Note: roughSegment.coords[0] is last point added
            # Note: AstarSearch3D usesformat of (z, y, z)
            logger.info(f"x1 {x1} x2 {x2} y1 {y1} y2 {y2}")
            # last Point = (x1, y1), 1st point = (x2,y2)

            if x1 < x2 and y1 < y2: # ex: z = 32 added first: x2,y2 = (811 939) added last: x1,y1 = (729 908)
                logger.info(f"case 1")
                astar = brightest_path_lib.algorithm.AStarSearch(image, 
                    start_point = np.array([reIndexZ,0,0]), 
                    goal_point = np.array([reIndexZ,newImageSize[0],newImageSize[1]]))

            elif x1 > x2 and y1 > y2: # ex: z = 32 x2,y2 = (729 908) x1,y1 = (811 939)
                logger.info(f"case 2")
                astar = brightest_path_lib.algorithm.AStarSearch(image, 
                    start_point = np.array([reIndexZ,newImageSize[0],newImageSize[1]]), 
                    goal_point = np.array([reIndexZ,0,0]))
                
            elif x1 > x2 and y1 < y2: # ex: z = 11 x2,y2 = (209, 168) x1,y1 = (250, 143) 
                logger.info(f"case 3")
                astar = brightest_path_lib.algorithm.AStarSearch(image, 
                        start_point = np.array([reIndexZ,0,newImageSize[1]]), # bigger y ending
                        goal_point = np.array([reIndexZ,newImageSize[0],0])) # bigger x starting
                
            elif x1 < x2 and y1 > y2: # ex: z = 11  x2,y2 = (250, 143) x1,y1 = (209, 168)
                logger.info(f"case 4")
                astar = brightest_path_lib.algorithm.AStarSearch(image, 
                    start_point = np.array([reIndexZ,newImageSize[0],0]), 
                    goal_point = np.array([reIndexZ,0,newImageSize[1]])) 
                
            elif x1 < x2 and y1 == y2: # ex: z = 20 (593,401) (584, 401))
                logger.info(f"case 5")
                # y = 0 = newImageSize[0]
                astar = brightest_path_lib.algorithm.AStarSearch(image, 
                    start_point = np.array([reIndexZ,0,0]),
                    goal_point = np.array([reIndexZ,0,newImageSize[1]]))
                
            elif x1 > x2 and y1 == y2: # ex: z = 20 (584, 401) (593,401) 
                logger.info(f"case 6")
                astar = brightest_path_lib.algorithm.AStarSearch(image, 
                    start_point = np.array([reIndexZ,0,newImageSize[1]]), 
                    goal_point = np.array([reIndexZ,0,0])) 

            elif x1 == x2 and y1 > y2: # ex: z = 25 (237,614) (237, 631))
                logger.info(f"case 7")
                astar = brightest_path_lib.algorithm.AStarSearch(image, 
                    start_point = np.array([reIndexZ,newImageSize[0],0]), 
                    goal_point = np.array([reIndexZ,0,0])) 
                
            elif x1 == x2 and y1 < y2: # ex: z = 25 (237, 631)) (237,614) 
                logger.info(f"case 8")
                astar = brightest_path_lib.algorithm.AStarSearch(image, 
                    start_point = np.array([reIndexZ,0,0]), 
                    goal_point = np.array([reIndexZ,newImageSize[0],0])) 
                
            elif x1 == x2 and y1 == y2:
                logger.info(f"case 9")
                # Should not enter here
                logger.error(f"same point should not be clicked twice")
            
            # Standard Search with full image
            # astar = brightest_path_lib.algorithm.AStarSearch(image, start_point = np.array([reIndexZ,y1,x1]), 
            #                                     goal_point = np.array([reIndexZ,y2,x2]))
            
            path = astar.search()

            # z = z value passed in, imageZ is the index of the image (that was reset), reIndexZ is the median of the new indexes
            # translate new image indexing back to our original image shape
            # For: Bounding Box 3D
            logger.info(f"z {z} imageZ {imageZ} reIndexZ {reIndexZ}")
            path = np.array([[x + x_min, y + y_min, z + imageZ - reIndexZ] for imageZ, y, x in LineString(path).coords]) 

            # Using B-Spine Interpolation for smoothing
            x, y, z = path[:, 0], path[:, 1], path[:, 2]
            tck, u = splprep([x, y, z], s=3)  # Adjust s for smoothing
            # For Improvement: could set num to be a ratio of pixels (distance between points) and a constant
            u_new = np.linspace(0, 1, num=5)  # Reduce points
            x_smooth, y_smooth, z_smooth = splev(u_new, tck)

            # maintain original integer form rather than float
            x_smooth = np.round(x_smooth).astype(int)
            y_smooth = np.round(y_smooth).astype(int)
            z_smooth = np.round(z_smooth).astype(int)

            # ensure that first point is the same as original
            x_smooth[0], y_smooth[0], z_smooth[0] = x[0], y[0], z[0]

            smoothedData = list(zip(x_smooth, y_smooth, z_smooth))
            lineStr = LineString(smoothedData)
        
            logger.info(f"lineStr {lineStr}")

            return lineStr
        
        return roughSegment

    # DEFUNCT - original implementation
    def old_brightestPath(self, roughSegment: LineString, live: bool = False, z: int = None):
        """
        Args:
            roughSegment:  LineString([left, point]) or LineString([point, right])
            live: if live use brightest path tracing to return a set of points from roughSegment[0] to roughSegment[1]
            z: current z Slice
        """
        logger.info(f"roughSegment coming in {roughSegment}")
        # if live:
        #     # TODO: Consider adding the mutation type along with the prior result if we can use it to speed things up
        #     return None

        zSpread = self.analysisParams.getValue('zSpread')
        channel = self.analysisParams.getValue('channel')

        # logger.info(f"z: {z} zSpread: {zSpread} channel {channel}")

        # 3D
        image = self.getPixels(channel=channel, z=z, zSpread=zSpread, threeD = True).data(flattened=False) # returning in ndarray form

        x1,y1,z1 = roughSegment.coords[0]
        x2,y2,z2 = roughSegment.coords[1]
        # logger.info(f"roughSegment.coords[0] {roughSegment.coords[0]}")
        # logger.info(f"roughSegment.coords[1] {roughSegment.coords[1]}")
        
        # For 3D:
        # get a median index since image does not retain original segment row indexes
        imageZ, imageX, imageY = image.shape  # (z, x, y)
        reIndexZ = math.floor(imageZ/2)
        logger.info(f"reIndexZ {reIndexZ}")


        # logger.info(f"new Image {image}")
        if live:

            # After testing, I found start_point and goal_point order matters for A/NBAStarSearch
            # start_point should always be the last point added (appended)
            # Note: roughSegment[0] is the last point added
            # accounting for the two cases:

            # Standard Search with full image
            start_point = np.array([reIndexZ,y1,x1])
            goal_point = np.array([reIndexZ,y2,x2])
            # abb
            # expecting (z, x, y)
            # start_point = np.array([reIndexZ,x1,y1])
            # goal_point = np.array([reIndexZ,x2,y2])
            logger.warning(f'AStarSearch with image:{image.shape} start_point:{start_point} goal_point:{goal_point}')
            astar = brightest_path_lib.algorithm.AStarSearch(image, start_point = start_point, 
                                                goal_point = goal_point
                                                )
            
            path = astar.search()

            # Standard Reformatting of Coordinates
            # Note: AStar Path Coords are (z,y,x), while our Linestrings are (x,y,z)
            path = np.array([[x, y, z + imageZ - reIndexZ] for imageZ, y, x in LineString(path).coords]) # 3D

            # Using B-Spine Interpolation for smoothing
            x, y, z = path[:, 0], path[:, 1], path[:, 2]
            tck, u = splprep([x, y, z], s=3)  # Adjust s for smoothing
            # For Improvement: could set num to be a ratio of pixels (distance between points) and a constant
            u_new = np.linspace(0, 1, num=5)  # Reduce points
            x_smooth, y_smooth, z_smooth = splev(u_new, tck)

            # maintain original integer form rather than float
            x_smooth = np.round(x_smooth).astype(int)
            y_smooth = np.round(y_smooth).astype(int)
            z_smooth = np.round(z_smooth).astype(int)

            smoothedData = list(zip(x_smooth, y_smooth, z_smooth))
            lineStr = LineString(smoothedData)
        
            logger.info(f"lineStr {lineStr}")

            return lineStr
        
        return roughSegment
