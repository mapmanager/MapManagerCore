import numpy as np
from .layer import Layer

class PolygonLayer(Layer):
    def _encodeBin(self):
        featureId = self.series.index
        coords = self.series
        coords = coords.reset_index(drop=True)
        polygonIndices = coords.count_coordinates().cumsum()
        coords = coords.get_coordinates()

        return {"polygons": {
            "ids": featureId,
            "featureIds": coords.index.to_numpy(dtype=np.uint16),
            "polygonIndices": np.insert(polygonIndices.to_numpy(dtype=np.uint16), 0, 0, axis=0),
            "positions": coords.to_numpy(dtype=np.float32).flatten(),
        }}
