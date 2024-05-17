from io import BytesIO, StringIO
import json
from typing import Tuple

import numpy as np

from ..benchmark import timeAll
from ..config import SpineId
from .single_time_point.layers import AnnotationsOptions
from ..image_slices import ImageSlice
from ..layers.utils import inRange
from ..loader.mmap import MMapLoader
from ..utils import filterMask
from . import Annotations
from pyodide.http import pyfetch
from pyodide.ffi import to_js
from .single_time_point import SingleTimePointAnnotations


class PyodideSingleTimePoint(SingleTimePointAnnotations):
    @timeAll
    def getAnnotations_js(self, options: AnnotationsOptions):
        """
        A JS version of getAnnotations.
        """
        options = options.to_py()
        layers = self.getAnnotations(options)
        return [layer.encodeBin() for layer in layers]

    def metadata_json(self):
        return json.dumps(self.metadata())

    def getSpinePosition(self, spineID: SpineId):
        if (spineID, self._t) not in self._annotations._points.index:
            return None
        return to_js(list(self._annotations._points.loc[(spineID, self._t), "point"].coords)[0])

    def getSegmentsAndSpines(self, options: AnnotationsOptions):
        options = options.to_py()
        z_range = options['zRange']
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

    def slices_js(self, channel: int, zRange: Tuple[int, int]) -> ImageSlice:
        """
        Loads the image data for a slice.

        Args:
          channel (int): The channel index.
          zRange ([int, int]): The visible z slice range.

        Returns:
          ImageSlice: The image slice.
        """
        return self.getPixels(channel, (zRange[0], zRange[1]))


class PyodideAnnotations(Annotations):
    """ PyodideAnnotations contains pyodide specific helper methods to allow JS to use Annotations.
    """

    async def load(path: str):
        loader = MMapLoader(path)
        return PyodideAnnotations(loader)

    def timePoint_js(self, time: int):
        return PyodideSingleTimePoint(self, time)

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
    
    def getColumn(self, column: str):
        result = self[column].to_list()
        if len(result) > 1:
            if isinstance(result[0], np.integer):
                return [int(i) for i in result]
        return result
    
    def columnsAttributes_json(self): 
        return json.dumps(self.columnsAttributes)


async def loadGeoCsv(path):
    response = await pyfetch(path)
    csv_text = await response.text()
    return StringIO(csv_text)


async def fetchBytes(url: str):
    response = await pyfetch(url)
    return BytesIO(await response.memoryview())
