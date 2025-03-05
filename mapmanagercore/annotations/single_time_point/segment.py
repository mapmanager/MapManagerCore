import math
from typing import Union

import brightest_path_lib.algorithm
from matplotlib import pyplot as plt
import numpy as np
import scipy
from scipy.interpolate import splprep, splev
from mapmanagercore.utils import injectLine
from .base import SingleTimePointAnnotationsBase
from shapely.geometry import LineString, Point
import brightest_path_lib
from mapmanagercore.logger import logger

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
            if roughSegment.coords[0] == roughSegment.coords[1]: # ensure same point is not clicked twiced
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
        imageZ, imageX, imageY = image.shape
        reIndexZ = math.floor(imageZ/2)
        logger.info(f"reIndexZ {reIndexZ}")

        # Testing Restriction of image to a bounding box
        # boundingBoxRange = 0
        # x_min, x_max = int(min(x1,x2) - boundingBoxRange), int(max(x1,x2) + boundingBoxRange) # X-axis
        # y_min, y_max = int(min(y1,y2) - boundingBoxRange), int(max(y1,y2) + boundingBoxRange) # Y-axis

        # # Restrict image to a bounding box of rough Segment
        # # TODO: account for when x and y max arent the same point
        # image = image[:, y_min:y_max+1, x_min:x_max+1]
        # height, width = image.shape[1], image.shape[2]  # Get Y (height) and X (width)
        # top_left = (0, 0)  # Always starts at (Y=0, X=0)
        # newImageSize = (height - 1, width - 1)  # Last Y and X index
  
        # logger.info(f"new Image {image}")
        if live:
            # 3D  (z, x, y)

            # After testing, I found start_point and goal_point order matters for A/NBAStarSearch
            # start_point should always be the last point added (appended)
            # Note: roughSegment[0] is the last point added
            # accounting for the two cases:
            logger.info(f"roughSegment.coords[0] {roughSegment.coords[0]}")
            logger.info(f"roughSegment.coords[1] {roughSegment.coords[1]}")

            # For bounding box image: (WIP)
            # Currently failing edge case where X1 > X2 but Y1 < Y2
            # if roughSegment.coords[0] > roughSegment.coords[1]: 
            #     logger.info(f"case 1")
            #     astar = brightest_path_lib.algorithm.AStarSearch(image, 
            #             start_point = np.array([reIndexZ,newImageSize[0],newImageSize[1]]), 
            #             goal_point = np.array([reIndexZ,0,0]))
                
            # else: # handles case: roughSegment.coords[0] < roughSegment.coords[1] and ==
            #     logger.info(f"case 2")
            #     astar = brightest_path_lib.algorithm.AStarSearch(image, 
            #             start_point = np.array([reIndexZ,0,0]), 
            #             goal_point = np.array([reIndexZ,newImageSize[0],newImageSize[1]]))
            
            # Standard Search with full image
            astar = brightest_path_lib.algorithm.AStarSearch(image, start_point = np.array([reIndexZ,y1,x1]), 
                                                goal_point = np.array([reIndexZ,y2,x2]))
            
            path = astar.search()

            # # z being the z value passed in, imageZ is the index of the image (that was resetted), reIndexZ is the median of the new indexes
            # # translate this back into our original image shape
            # For: Bounding Box 3D
            # path = np.array([[x + x_min, y + y_min, z + imageZ - reIndexZ] for imageZ, y, x in LineString(path).coords]) 

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
