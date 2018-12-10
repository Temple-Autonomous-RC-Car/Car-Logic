import numpy as np
import cv2


def flatten_perspective(image):
    """
    Warps the image from the vehicle front-facing camera mapping hte road to a bird view perspective.

    Parameters
    ----------
    image       : Image from the vehicle front-facing camera.

    Returns
    -------
    Warped image.
    """
    # Get image dimensions
    (h, w) = (image.shape[0], image.shape[1])
    # Define source points
    source = np.float32([[w // 2 - w*.30, h * .6], [w // 2 + w*.30, h * .6], [0 + .1*w, h], [w - .1*w, h]])
    # Define corresponding destination points
    destination = np.float32([[100, 0], [w - 100, 0], [100, h], [w - 100, h]])
    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    unwarp_matrix = cv2.getPerspectiveTransform(destination, source)
    return (cv2.warpPerspective(image, transform_matrix, (w, h)), unwarp_matrix)