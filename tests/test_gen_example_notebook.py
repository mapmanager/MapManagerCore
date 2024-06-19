import matplotlib
import unittest
import os

class TestExamplesNotebook(unittest.TestCase):

    def test_notebook(self):
        wd = os.curdir
        os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../examples/'))
        matplotlib.use('Agg')
        try:
            from mapmanagercore import MapAnnotations, MultiImageLoader, MMapLoader
            import matplotlib.pyplot as plt
            # Create an image loader
            loader = MultiImageLoader(
                lineSegments="../data/rr30a_s0u/line_segments.csv",
                points="../data/rr30a_s0u/points.csv",
                metadata="../data/rr30a_s0u/metadata.json",)
            
            # add image channels to the loader
            loader.read("../data/rr30a_s0u/t0/rr30a_s0_ch1.tif", channel=0)
            loader.read("../data/rr30a_s0u/t0/rr30a_s0_ch2.tif", channel=1)
            
            loader.read("../data/rr30a_s0u/t0/rr30a_s0_ch1.tif", channel=0, time=1)
            loader.read("../data/rr30a_s0u/t0/rr30a_s0_ch2.tif", channel=1, time=1)
            
            # Create the annotation map
            map = MapAnnotations(loader)
            
            # map.addSegment(t=0)
            
            # save the annotation map
            map.save("../data/rr30a_s0u.mmap")
            map.metadata()
            map.images.shape()
            map.columns
            # loading the map manager from zarr.
            map = MapAnnotations(MMapLoader("../data/rr30a_s0u.mmap").cached())
            map._points.columns
            xySpine = map._points['point'].get_coordinates()
            xyAnchor = map._points['anchor'].get_coordinates()
            
            map._points.columns
            xySpine['z'] = map._points['z']
            xySpine['segmentID'] = map._points['segmentID']
            xySpine['userType'] = map._points['userType']
            xySpine['accept'] = map._points['accept']
            xySpine['note'] = map._points['note']
            
            xySpine['anchorX'] = xyAnchor['x']
            xySpine['anchorY'] = xyAnchor['y']
            xySpine
            map[:].columns
            map.segments["segment"].get_coordinates(include_z=True)
            # spine df for tp 0
            filtered = map[ map['t']==0 ]
            filtered[:]
            sessionID = 0
            spineID = 4
            map.deleteSpine((id, sessionID))
            id = map.addSpine(segmentId=(0, 0), x=1,y=2,z=3)
            # map.moveAnchor(spineId=id, x=1, y=1, z=3)
            # map.moveSpine(spineId=id, x=1, y=1, z=3)
            # map.deleteSpine((id, sessionID)
            # map.undo()
            # map.redo()
            # map.updateSpine(spineId=id, value={
            #   "f": 1,
            # })
            # map.undo()
            # map.translateBackgroundRoi()
            # map.deleteSegment("")
            
            map.updateSpine(spineId=1, value={
              "note": "This is a note",
            })
            map[1, "note"]
            map.segments["segment"].get_coordinates(include_z=True)
            map.segments["segmentLeft"].get_coordinates()
            map.segments["segmentRight"].get_coordinates(include_z=True)
            filtered = map[ map['t']==1 ]
            filtered["roi"].get_coordinates()
            map["roiBase"].get_coordinates()
            slices = map.getPixels(time=0, channel=0, zRange=(18, 36))
            plt.hist(slices.data(), bins=50)
            plt.yscale('log')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.title('Histogram of Image Data')
            plt.show()
            
            fig, ax = plt.subplots(figsize=(10, 10))
            
            map.segments["segmentLeft"].plot(color='blue', linestyle='dotted', ax=ax)
            map.segments["segmentRight"].plot(color='green', linestyle='dotted', ax=ax)
            
            map["anchors"].plot(color='black', ax=ax)
            map["points"].plot(color='red', marker='o', markersize=2, ax=ax)
            
            map["roiHead"].plot(edgecolor='blue', color=(0,0,0,0), ax=ax)
            map["roiHeadBg"].plot(edgecolor='blue', linestyle='dotted', color=(0,0,0,0), ax=ax)
            
            map["roiBase"].plot(edgecolor='red', color=(0,0,0,0), ax=ax)
            map["roiBaseBg"].plot(edgecolor='red', linestyle='dotted', color=(0,0,0,0), ax=ax)
            
            slices.plot(ax=ax, vmin=300, vmax=1500, alpha=0.5, cmap='CMRmap')
            
            plt.show()
            map["z"].between(10, 40)
            filtered = map[map["z"].between(10, 40)]
            map[:]
            filtered[:]
            slices = filtered.getPixels(time=0, channel=0)
            
            fig, ax = plt.subplots(figsize=(10, 10))
            
            filtered.segments["segmentLeft"].plot(color='blue', linestyle='dotted', ax=ax)
            filtered.segments["segmentRight"].plot(color='green', linestyle='dotted', ax=ax)
            
            filtered["anchors"].plot(color='black', ax=ax)
            filtered["points"].plot(color='red', marker='o', markersize=2, ax=ax)
            
            filtered["roi"].plot(edgecolor='blue', color=(0,0,0,0), ax=ax)
            filtered["roiBg"].plot(edgecolor='red', linestyle='dotted', color=(0,0,0,0), ax=ax)
            
            slices.plot(ax=ax, vmin=300, vmax=1500, alpha=0.45, cmap='gray')
            
            # Set x and y limits
            ax.set_xlim(300, 800)
            ax.set_ylim(600, 200)
            
            plt.show()
            layers = filtered.getAnnotations(options={
                "selection": {
                  "t": 0,
                  "z": (18, 36)
                },
                "annotationSelections": {
                  "segmentIDEditing": None,
                  "segmentID": None,
                  "spineID": None
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