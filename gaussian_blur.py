from __future__ import annotations
import numpy as np
from scipy.ndimage import convolve
from utils.image_utils import load_image, save_image
from dataclasses import dataclass
import cv2


@dataclass
class GaussianFilterParams:
    kernel_size: int
    var: float


def _gaussian_kernel(size: int, var: float) -> np.ndarray:
    """Generates a gaussian filter

    Args:
        size (int): size of the kernel (size x size)
        var (float): variance of the kernel

    Returns:
        np.ndarray: the gaussian kernel
    """
    # Initialize kernel matrix
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    factor = 1 / (2 * np.pi * var)

    # Populate kernel values
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = factor * np.exp(-(x**2 + y**2) / (2 * var))

    # Normalize kernel
    kernel /= np.sum(kernel)

    return kernel


def estimate_gaussian_params(img_size: tuple) -> GaussianFilterParams:
    # Get image width and height
    if len(img_size) == 3:
        img_height, img_width, _ = img_size
    else:
        img_height, img_width = img_size

    # Estimate the appropriate kernel size
    kernel_size = min(img_height // 2, img_width // 2)
    kernel_size = (
        kernel_size if (kernel_size % 2) == 1 else (kernel_size + 1)
    )  # must be odd

    # Estimate sigma
    var = (kernel_size**2) // 25

    return GaussianFilterParams(kernel_size=kernel_size, var=var)


def create_gaussian_kernel(img_size: tuple) -> np.ndarray:
    params = estimate_gaussian_params(img_size)

    return _gaussian_kernel(size=params.kernel_size, var=params.var)


def apply_gaussian_blur(image: np.ndarray, pt1: tuple, pt2: tuple) -> np.ndarray:
    """Return the convolution result: image * kernel.

    Args:
        image (np.ndarray): image of size H x W x C
        kernel (np.ndarray): kernel of size h x w

    Returns:
        np.ndarray: convolution result of the shape H x W x C
    """
    # Get points
    x1, y1 = pt1
    x2, y2 = pt2

    # Generate a Gaussian kernel
    box_size = (abs(x2 - x1), abs(y2 - y1))
    kernel = create_gaussian_kernel(img_size=box_size)

    # Ensure the input image has three channels
    channels = len(image.shape)
    if channels != 3:
        raise Warning("WARNING: Input image is not a 3-channel image.")

    # Area to be convolved
    box_img = image[y1:y2, x1:x2, :]

    blurred_box = [
        convolve(input=box_img[:, :, c], weights=kernel, mode="reflect")
        for c in range(channels)
    ]
    blurred_box = np.stack(blurred_box, axis=-1)
    image[y1:y2, x1:x2, :] = blurred_box

    return image


def apply_gaussian_blur_opencv(frame):
    params = estimate_gaussian_params(frame.shape)
    kernel_size = params.kernel_size
    sigma = np.sqrt(params.var)

    return cv2.GaussianBlur(
        src=frame, ksize=(kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma
    )


# Test script
if __name__ == "__main__":
    # Example 3-channel image 
    image = load_image(image_path="./face_dataset/images/train/00a0d634ad200ced.jpg")
    # Random window
    pt1 = (10, 50)
    pt2 = (400, 600)

    # Convolve the image with the Gaussian kernel
    convolved_image = apply_gaussian_blur(image, pt1, pt2)
    save_image(convolved_image, "blurred_img.jpg")
