
from mapmanagercore import MapAnnotations, MultiImageLoader
import mapmanagercore.data
from mapmanagercore.lazy_geo_pd_images.loader.base import ImageLoader
# from ..lazy_geo_pd_images import LazyImagesGeoPandas, ImageLoader

loader = MultiImageLoader()

path_ch1 = mapmanagercore.data.getTiffChannel_1()

loader.read(path_ch1, channel=0)
_build : ImageLoader = loader
map = MapAnnotations(_build)

print("total channels", map._channels())
# try and add a second channel to map
# we need to add the second channel to LazyImagesGeoPandas._images

# -----------------Load 2nd channel ---------------------
path_ch2 = mapmanagercore.data.getTiffChannel_2()
# loader.read(path_ch2, channel=1)  # ????
# _build : ImageLoader = loader.build()
# map = MapAnnotations(_build)
# -----------------Load 2nd channel ---------------------

# print("total channels", map._channels())

# need something like
# MapAnnotations will need to call its imageloader to read
# current problem mapannotations has access to imageloader but not the inherited multiimageloader
map.loadInNewChannel(path = path_ch2, channel=1)

print("total channels again: ", map._channels())



# general problem is that this was all designed to be pre-built with all desired channels
# we need it to work 'from scratch'

# how can we check if channels are actually loaded?
