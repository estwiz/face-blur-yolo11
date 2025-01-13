import os
import random

from utils.plot_utils import plot_images_with_boxes

if __name__ == "__main__":
    BASE_DIR = "./face_dataset/images/train"

    # Select random images
    all_images = [
        os.path.join(BASE_DIR, img)
        for img in os.listdir(BASE_DIR)
        if img.endswith(".jpg")
    ]
    selected_images = random.sample(all_images, min(9, len(all_images)))

    # Plot images with bounding boxes
    plot_images_with_boxes(selected_images)
