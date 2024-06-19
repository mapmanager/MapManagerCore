from mapmanagercore.lazy_geo_pd_images.metadata import Metadata
from .base import ImageLoader
from typing import Iterator
import numpy as np
import zarr


class ZarrLoader(ImageLoader):
    """A loader for images stored in a zarr file."""

    def __init__(self, path: str, lazy: bool = False):
        """
        Initializes a ZarrLoader object.

        Args:
            path (str): The path to the Zarr file.
            lazy (bool, optional): If True, the images will be loaded lazily. 
                If False, the images will be loaded eagerly. Defaults to False.
        """
        super().__init__()

        self.store = zarr.ZipStore(path, mode="r")
        self.group = zarr.group(store=self.store)
        self._imagesSrcs = {}
        self._metadata = {}
        for t in self.group.attrs["timePoints"]:
            images = self.group[f"img-{t}"]
            self._imagesSrcs[t] = images if lazy else images[:]
            self._metadata[t] = Metadata.from_json(self.group.attrs[f"metadata-{t}"])
            
        self.path = path

    def __str__(self):
        return f"Zarr Loader: path: {self.path}"

    def timePoints(self) -> Iterator[int]:
        """
        Returns an iterator over the time points of the images.

        Returns:
            An iterator that yields the time points of the images.
        """
        return self._imagesSrcs.keys()

    def _images(self, t: int) -> np.ndarray:
        return self._imagesSrcs[t]

    def close(self):
        self.store.close()
