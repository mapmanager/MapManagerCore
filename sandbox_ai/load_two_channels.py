# abai 20250806: High-level workflow checking frame columns after each channel addition

from mapmanagercore.lazy_geo_pd_images.loader.mm_map_loader import mmMapLoader
from mapmanagercore import MapAnnotations
import pandas as pd

# 1. Create loader and import first timepoint (with channel 1)
loader = mmMapLoader()
timepoint_path = 'data/rr30a_s0u/t0/rr30a_s0_ch1.tif'
timepoint_key = loader.importTimepoint(timepoint_path)  # abai 20250806

# 2. Create MapAnnotations object
map = MapAnnotations(loader,
                     lineSegments=pd.DataFrame(),
                     points=pd.DataFrame())  # abai 20250806

# 3. Print current channel keys and frame columns
# print("After adding channel 1, channel keys:", loader.metadata.getTimepoint(timepoint_key).channelKeys)  # abai 20250806
# print("Segment columns after channel 1:", map.segments.columns)  # abai 20250806
print("Point columns after channel 1:", map.points.columns)  # abai 20250806

# 4. Add second channel to the same timepoint
channel2_path = 'data/rr30a_s0u/t0/rr30a_s0_ch2.tif'
loader.importChannel(channel2_path, timepoint_key)  # abai 20250806

# 5. Print updated channel keys and frame columns
# print("After adding channel 2, channel keys:", loader.metadata.getTimepoint(timepoint_key).channelKeys)  # abai 20250806
# print("Segment columns after channel 2:", map.segments.columns)  # abai 20250806
print("Point columns after channel 2:", map.points.columns)  # abai 20250806