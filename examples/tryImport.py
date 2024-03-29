from mapmanagercore import MapAnnotations, MultiImageLoader, MMapLoader
import matplotlib.pyplot as plt

# Create an image loader
loader = MultiImageLoader(
    lineSegments="../data/rr30a_s0u/line_segments.csv",
    points="../data/rr30a_s0u/points.csv")

# add image channels to the loader
loader.read("../data/rr30a_s0u/t0/rr30a_s0_ch1.tif", channel=0)
loader.read("../data/rr30a_s0u/t0/rr30a_s0_ch2.tif", channel=1)

# Create the annotation map
map = MapAnnotations(loader)