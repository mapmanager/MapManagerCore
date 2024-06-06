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
        if (spineID, self._t) not in self._annotations.points.index:
            return None
        return to_js(list(self._annotations.points[(spineID, self._t), "point"].coords)[0])

    def getSegmentsAndSpines(self, options: AnnotationsOptions):
        options = options.to_py()
        z_range = options['zRange']
        index_filter = options["filters"] if "filters" in options else []
        segments = []
        missingSegments = set(self.segments.index)
        for (segmentID, points) in self.points[["segmentID", "z"]].groupby("segmentID"):
            missingSegments.remove(segmentID)
            spines = points.index.to_frame(name="id")
            spines["type"] = "Start"
            spines["invisible"] = ~ inRange(points["z"], z_range)
            spines["invisible"] = spines["invisible"] & ~ filterMask(
                points.index, index_filter)

            segments.append({
                "segmentID": segmentID,
                "spines": spines.to_dict('records')
            })

        for segmentId in missingSegments:
            segments.append({
                "segmentID": segmentId,
                "spines": []
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

    def table(self):
        columns = [key for key, value in self.points.columnsAttributes.items() if value["plot"]]
        columns.remove("t")
        df = self.points[columns].reset_index()
        for i, type in enumerate(df.dtypes):
            if type == np.integer:
                df.iloc[:, i] = df.iloc[:, i].astype(int)
        return df

    def getColumn(self, column: str):
        if column in self.points.index.names:
            result = self.points.index.get_level_values(column).to_list()
        else:
            result = self.points[column].to_list()
        if len(result) > 1:
            if isinstance(result[0], np.integer):
                return [int(i) for i in result]
        return result

    def getColors(self, colorOn: str = None):
        return super().getColors(colorOn).to_list()
    
    def getSymbols(self, shapeOn: str = None):
        return super().getSymbols(shapeOn).to_list()

    def columnsAttributes_json(self):
        return json.dumps(self.points.columnsAttributes, skipkeys=True)


async def loadGeoCsv(path):
    response = await pyfetch(path)
    csv_text = await response.text()
    return StringIO(csv_text)


async def fetchBytes(url: str):
    response = await pyfetch(url)
    return BytesIO(await response.memoryview())
