import numpy as np

def _makeCheckerboard(imageShape: tuple[int, int, int],
                      squareSize: int,
                      intensity:int = 1) -> np.ndarray:
    """
    Create a checkerboard pattern in a 3D image.

    Each piece of the checkerboard goes through all imageShape[0] slices.

    Parameters
    ----------
    imageShape : Tuple[int, int, int]
        Shape of the 3D image (depth, height, width).
    squareSize : int
        Size of each square in the checkerboard pattern.
    intensity : int, optional
        Intensity value for the filled squares (default is 1).

    Returns
    -------
    arr : np.ndarray
        A 3D numpy array with the checkerboard pattern.
    """
    _depth, imageHeight, imageWidth = imageShape
    arr = np.zeros(imageShape, dtype=np.uint8)

    for i in range(0, imageHeight, squareSize):
        for j in range(0, imageWidth, squareSize):
            arr[:, i:i+squareSize, j:j+squareSize] = (
                0 if ((i // squareSize) + (j // squareSize)) % 2 == 0 else intensity
            )
    return arr

