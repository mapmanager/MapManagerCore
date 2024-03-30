from mapmanagercore import MapAnnotations, MultiImageLoader, MMapLoader
import matplotlib.pyplot as plt

# Create an image loader
loader = MultiImageLoader(
    lineSegments="../data/rr30a_s0u/line_segments.csv",
    points="../data/rr30a_s0u/points.csv")

sys.exit(1)

# add image channels to the loader
loader.read("../data/rr30a_s0u/t0/rr30a_s0_ch1.tif", channel=0)
loader.read("../data/rr30a_s0u/t0/rr30a_s0_ch2.tif", channel=1)

# Create the annotation map
map = MapAnnotations(loader)

# save the annotation map
map.save("../data/rr30a_s0us.mmap")

map = MapAnnotations(MMapLoader("../data/rr30a_s0us.mmap").cached())

# print(map._table().head())

cols = map.columns
print(cols)

cols = ['x', 'y', 'z', 'segmentID', 'note', 'userType', 'spineLength']
df = map[cols]
print(df)
