import matplotlib
import unittest
import os

class TestExamplesNotebook(unittest.TestCase):

    def test_notebook(self):
        wd = os.curdir
        os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../examples/'))
        matplotlib.use('Agg')
        try:
            from mapmanagercore import MapAnnotations, MultiImageLoader
            from mapmanagercore.data import getLinesFile, getPointsFile, getTiffChannel_1, getTiffChannel_2
            import matplotlib.pyplot as plt
            # Create an image loader
            loader = MultiImageLoader()
            
            # add image channels to the loader
            loader.read(getTiffChannel_1(), channel=0)
            loader.read(getTiffChannel_2(), channel=1)
            
            # loader.read("../data/rr30a_s0u/t0/rr30a_s0_ch1.tif", channel=0, time=1)
            # loader.read("../data/rr30a_s0u/t0/rr30a_s0_ch2.tif", channel=1, time=1)
            
            
            # Create the annotation map
            lineSegments=getLinesFile()
            print('lineSegments:', lineSegments)
            map = MapAnnotations(loader.build(),
                                # lineSegments="../data/rr30a_s0u/line_segments.csv",
                                # points="../data/rr30a_s0u/points.csv",
                                lineSegments=getLinesFile(),
                                points=getPointsFile()
                                )
            map.points[:]
            map.segments[:]
            
            # save the annotation map
            map.save("../data/rr30a_s0u.mmap")
            map.close()
            map = MapAnnotations.load("../data/rr30a_s0u.mmap")
            fig, ax = plt.subplots(figsize=(5, 5))
            
            map.points["anchorLine"].plot(color='black', ax=ax)
            map.points["point"].plot(color='red', marker='o', markersize=2, ax=ax)
            
            map.points["roiHead"].plot(edgecolor='blue', color=(0,0,0,0), ax=ax)
            map.points["roiHeadBg"].plot(edgecolor='blue', linestyle='dotted', color=(0,0,0,0), ax=ax)
            
            map.points["roiBase"].plot(edgecolor='red', color=(0,0,0,0), ax=ax)
            map.points["roiBaseBg"].plot(edgecolor='red', linestyle='dotted', color=(0,0,0,0), ax=ax)
            
            slices = map.getPixels(time=0, channel=0, zRange=(18, 36))
            slices.plot(ax=ax, vmin=300, vmax=1500, alpha=0.5, cmap='CMRmap')
            
            plt.show()
            map.points[["roiStatsBg_ch1_max", "roiStatsBg_ch2_max", "roiStats_ch1_max", "roiStats_ch2_max"]]
            # map.points[:]
            timePoint = map.getTimePoint(0)
            id = timePoint.addSpine(segmentId=0, x=1,y=2,z=3)
            # timePoint.moveAnchor(spineId=id, x=1, y=1, z=3)
            # timePoint.moveSpine(spineId=id, x=1, y=1, z=3)
            # timePoint.deleteSpine(id)
            # timePoint.undo()
            # timePoint.redo()
            # timePoint.updateSpine(spineId=id, value={
            #   "f": 1,
            # })
            # timePoint.undo()
            # timePoint.translateBackgroundRoi()
            # timePoint.deleteSegment("")
            id
            
            # abb
            # timePoint.points["roi"].get_coordinates()
            # timePoint.points["roiBase"].get_coordinates()['x'].tolist()
            
            from mapmanagercore.schemas.spine import Spine
            
            
            map.updateSpine(spineId=1, value=Spine(
              note="This is a note",
            ))
            map.points[(1, ), ["note"]]
            timePoint.points[1, "note"]
            map.segments["segment"].get_coordinates(include_z=True)
            # map.segments["left"]
            map.points["roi"].get_coordinates()
            map.points['anchorLine']
            map.points["roiBase"].get_coordinates()
            slices = map.getPixels(time=0, channel=0, zRange=(18, 36))
            plt.hist(slices.data(), bins=50)
            plt.yscale('log')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.title('Histogram of Image Data')
            plt.show()
            
            fig, ax = plt.subplots(figsize=(10, 10))
            
            map.points["anchorLine"].plot(color='black', ax=ax)
            map.points["point"].plot(color='red', marker='o', markersize=2, ax=ax)
            
            map.points["roiHead"].plot(edgecolor='blue', color=(0,0,0,0), ax=ax)
            map.points["roiHeadBg"].plot(edgecolor='blue', linestyle='dotted', color=(0,0,0,0), ax=ax)
            
            map.points["roiBase"].plot(edgecolor='red', color=(0,0,0,0), ax=ax)
            map.points["roiBaseBg"].plot(edgecolor='red', linestyle='dotted', color=(0,0,0,0), ax=ax)
            
            slices.plot(ax=ax, vmin=300, vmax=1500, alpha=0.5, cmap='CMRmap')
            
            plt.show()
            map.points["z"].between(10, 40)
            filtered = map.filterPoints(map.points["z"].between(10, 40))
            filtered.points[:, "z"]
            slices = filtered.getPixels(time=0, channel=0)
            
            fig, ax = plt.subplots(figsize=(10, 10))
            
            filtered.points["anchorLine"].plot(color='black', ax=ax)
            filtered.points["point"].plot(color='red', marker='o', markersize=2, ax=ax)
            
            filtered.points["roi"].plot(edgecolor='blue', color=(0,0,0,0), ax=ax)
            filtered.points["roiBg"].plot(edgecolor='red', linestyle='dotted', color=(0,0,0,0), ax=ax)
            
            slices.plot(ax=ax, vmin=300, vmax=1500, alpha=0.45, cmap='gray')
            
            # Set x and y limits
            ax.set_xlim(300, 800)
            ax.set_ylim(600, 200)
            
            plt.show()
            layers = map.getTimePoint(0).getAnnotations(options={
                "zRange": (18, 36),
                "annotationSelections": {
                  "segmentIDEditing": 1,
                  "segmentIDEditingPath": None,
                  "segmentID": 1,
                  "spineID": 1
                },
                "showLineSegments": True,
                "showAnchors": True,
                "showLabels": True,
                "showLineSegmentsRadius": True,
                "showSpines": True,
              },
            )
            
            for layer in layers:
                coords, props = layer.coordinates()
                print("Properties:", props, "\n coords:", coords.head(2), "\n\n")
            
        finally:
            os.chdir(wd)

if __name__ == '__main__':
    unittest.main()