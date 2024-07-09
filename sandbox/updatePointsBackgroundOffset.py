
import csv
import pandas as pd

from mapmanagercore import MapAnnotations, MultiImageLoader
from mapmanagercore.data import getLinesFile, getPointsFile, getTiffChannel_1, getTiffChannel_2
import matplotlib.pyplot as plt

def main():
    path = "C:\\Users\\johns\Documents\\GitHub\\MapManagerCore\data\\rr30a_s0u\\points.csv"

    df = pd.read_csv(path)
    # for index in df["spineID"]:


    # loop through each id
    # calculate new background offsets for x and y
    # store them into csv and save as new file
    # Create an image loader

    loader = MultiImageLoader()

    # add image channels to the loader
    loader.read(getTiffChannel_1(), channel=0)
    loader.read(getTiffChannel_2(), channel=1)

    # Create the annotation map
    map = MapAnnotations(loader.build(),
                        # lineSegments="../data/rr30a_s0u/line_segments.csv",
                        # points="../data/rr30a_s0u/points.csv",
                        lineSegments=getLinesFile(),
                        points=getPointsFile()
                        )
    
    # :<class 'mapmanagercore.annotations.single_time_point.layers.AnnotationsLayers'> s
    sessionID = 0
    _sessionMap = map.getTimePoint(sessionID)
    # print(type(_sessionMap))
    df = _sessionMap.points[:]
    temp1 = df["xBackgroundOffset"]

    print("before: ", temp1)
    # print(df.columns.tolist())
    for index in df.index:
        # print(index)
        _sessionMap.snapBackgroundOffset(index)

    newDF = map.points[:]
    temp2 = newDF["xBackgroundOffset"]
    print("after: ", temp2)

    # loop through points csv and save

    df = pd.read_csv(path)
    for index in df["spineID"]:
        df["xBackgroundOffset"][index] = newDF["xBackgroundOffset"][index]
        df["yBackgroundOffset"][index] = newDF["yBackgroundOffset"][index]

    df.to_csv('newPoints.csv', index=False)


if __name__ == '__main__':
    main()