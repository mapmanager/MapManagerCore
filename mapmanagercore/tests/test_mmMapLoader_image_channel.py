
from mapmanagercore.lazy_geo_pd_images.loader.mm_map_loader import mmMapLoader, ImageChannel

def test_image_channel():
    path = '/Users/cudmore/Desktop/sample_mmaps/zarLoader2.mmap'
    mapLoader = mmMapLoader(path)
    
    for timepoint in mapLoader.metadata.timepointKeys:
        for channel in mapLoader.metadata.getTimepoint(timepoint).channelKeys:
            ic = ImageChannel(mapLoader, timepoint, channel)

            print(ic.numSlices)
            print(ic._channelPath)

            imgData = ic.getSlice(5)
            print(imgData.shape)

            print(ic.numLoaded)

            imgVolume = ic.getVolume(6)
            print(imgVolume.shape)
            print(ic.numLoaded)

            imgVolume = ic.getVolume(5, 20)  # 20 - 5 + 1 slices
            print(imgVolume.shape)
            print(ic.numLoaded)

            ic.loadAllImageData()
            print(ic.numLoaded)

            ic.unloadAllImageData()
            print(ic.numLoaded)

            break

if __name__ == '__main__':
    test_image_channel()