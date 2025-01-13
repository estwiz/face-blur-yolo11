from PIL import Image
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """Loads an image from the specified file path and converts it to a NumPy array.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: A NumPy array representing the image.
    """
    image = Image.open(image_path).convert("RGB")
    return np.array(image)


def save_image(image: np.ndarray, output_path: str) -> None:
    """Saves a NumPy array as an image.

    Args:
        image (np.ndarray): A NumPy array representing the image.
        output_path (str): Path to save the image file.
    """
    img = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    img.save(output_path)
    print(f"Image saved at: {output_path}")


def get_roi(frame_blur: np.ndarray, pt1: tuple, pt2: tuple):
    x1, y1 = pt1
    x2, y2 = pt2
    roi = frame_blur[y1:y2, x1:x2, :]
    return roi


def modify_frame(
    frame: np.ndarray, blurred_roi: np.ndarray, pt1: tuple, pt2: tuple
) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    frame[y1:y2, x1:x2, :] = blurred_roi
