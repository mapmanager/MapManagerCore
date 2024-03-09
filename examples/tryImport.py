from mapmanagercore import MapAnnotations, ArrayImageLoaderBuilder
import matplotlib.pyplot as plt

# Create an image loader
imageLoader = ArrayImageLoaderBuilder()

# add image channels to the loader
imageLoader.read("../data/rr30a_s0u/t0/rr30a_s0_ch1.tif", channel=0)
imageLoader.read("../data/rr30a_s0u/t0/rr30a_s0_ch2.tif", channel=1)

# Create the annotation map
map = MapAnnotations(loader=imageLoader,
                     lineSegmentsPath="../data/rr30a_s0u/line_segments.csv",
                     pointsPath="../data/rr30a_s0u/points.csv")