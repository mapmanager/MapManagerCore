from io import BytesIO, StringIO
from typing import Tuple

from annotations.layers import AnnotationsOptions
from ..image_slices import ImageSlice
from ..layers.utils import inRange
from ..loader.imageio import MultiImageLoader
from .utils import filterMask
from .base import Annotations
from pyodide.http import pyfetch
from pyodide.ffi import to_js


class PyodideAnnotations(Annotations):
    """ PyodideAnnotations contains pyodide specific helper methods to allow JS to use Annotations.
    """

    async def load(url: str):
        lineSegments = await loadGeoCsv(url + "/line_segments.csv")
        points = await loadGeoCsv(url + "/points.csv")

        loader = MultiImageLoader(lineSegments=lineSegments, points=points)
        loader.read(await fetchBytes(url + "/t0/ch0.tif.br"), channel=0, time=0)
        loader.read(await fetchBytes(url + "/t0/ch1.tif.br"), channel=0, time=0)

        # TODO: Create a concurrent async Loader (subclass imageio's loader).

        return PyodideAnnotations(loader, lineSegments, points)

    def getAnnotations_js(self, options: AnnotationsOptions):
        """
        A JS version of getAnnotations.
        """
        options = options.to_py()
        layers = self.getAnnotations(options)
        return [layer.encodeBin() for layer in layers]

    def slices_js(self, time: int, channel: int, zRange: Tuple[int, int]) -> ImageSlice:
        """
        Loads the image data for a slice.

        Args:
          time (int): The time slot index.
          channel (int): The channel index.
          zRange ([int, int]): The visible z slice range.

        Returns:
          ImageSlice: The image slice.
        """
        return self.getPixels(time, channel, (zRange[0], zRange[1]))

    def getSpinePosition(self, t: int, spineID: SpineId):
        return to_js(list(self._points.loc[spineID, "point"].coords)[0])

    def getSegmentsAndSpines(self, options: AnnotationsOptions):
        options = options.to_py()
        z_range = options['selection']['z']
        index_filter = options["filters"] if "filters" in options else []
        segments = []

        for (segmentID, points) in self._points.groupby("segmentID"):
            spines = points.index.to_frame(name="id")
            spines["type"] = "Start"
            spines["invisible"] = ~ inRange(points["z"], z_range)
            spines["invisible"] = spines["invisible"] & ~ filterMask(
                points.index, index_filter)

            segments.append({
                "segmentID": segmentID,
                "spines": spines.to_dict('records')
            })

        return segments


async def loadGeoCsv(path, geometryCols, index_col=None, dtype=None):
    response = await pyfetch(path)
    csv_text = await response.text()
    return StringIO(csv_text)


async def fetchBytes(url: str):
    response = await pyfetch(url)
    return BytesIO(await response.memoryview())
