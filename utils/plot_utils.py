import uuid
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


def load_bounding_boxes(label_path: str, img_shape: tuple):
    """Load and scale bounding box coordinates."""
    try:
        with open(label_path, "r") as file:
            lines = [line.strip().split(" ")[1:] for line in file.readlines()]
        boxes = np.array(lines).astype(float)
    except FileNotFoundError:
        raise ValueError(f"Label file not found: {label_path}")

    # Scale boxes to image dimensions
    if len(img_shape) == 3:
        img_height, img_width, _ = img_shape
    else:
        img_height, img_width = img_shape
    
    boxes[:, 0::2] *= img_width
    boxes[:, 1::2] *= img_height

    # Convert to anchor format
    anchors = [
        [
            c_x - c_width / 2,  # Top-left x
            c_y - c_height / 2,  # Top-left y
            c_width,  # Width
            c_height,  # Height
        ]
        for c_x, c_y, c_width, c_height in boxes
    ]
    return anchors


def draw_boxes_on_image(img_path: str):
    """Draw bounding boxes on an image and return the figure, axes, and image data."""
    label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")

    # Load the image
    img = plt.imread(img_path)
    anchors = load_bounding_boxes(label_path, img.shape)

    # Create the plot
    fig, ax = plt.subplots()
    ax.imshow(img)

    # Add bounding boxes
    for anchor in anchors:
        rect = patches.Rectangle(
            (anchor[0], anchor[1]),  # Top-left corner
            anchor[2],  # Width
            anchor[3],  # Height
            linewidth=1,
            edgecolor="blue",
            facecolor="none",
        )
        ax.add_patch(rect)

    return fig, ax, img


def plot_images_with_boxes(image_paths: list, save_path="plots"):
    """Plot images with bounding boxes in a grid layout."""
    n_images = len(image_paths)
    grid_size = int(np.ceil(np.sqrt(n_images)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(6, 6))
    axes = axes.flatten()

    for idx, img_path in enumerate(image_paths):
        try:
            _, temp_ax, img = draw_boxes_on_image(img_path)

            # Display image in the subplot
            axes[idx].imshow(img)
            axes[idx].axis("off")  # Hide axes

            # Copy rectangles from the temporary Axes to the subplot
            for rect in temp_ax.patches:
                new_rect = patches.Rectangle(
                    rect.get_xy(),
                    rect.get_width(),
                    rect.get_height(),
                    linewidth=rect.get_linewidth(),
                    edgecolor=rect.get_edgecolor(),
                    facecolor=rect.get_facecolor(),
                    linestyle=rect.get_linestyle(),
                )
                axes[idx].add_patch(new_rect)

        except Exception as e:
            print(e)
        finally:
            plt.close()  # Close the temporary figure to free resources

    # Hide unused subplots
    for ax in axes[len(image_paths):]:
        ax.axis("off")

    plt.tight_layout()

    # Save image
    os.makedirs(save_path, exist_ok=True)
    plot_path = f"./{save_path}/fig_{uuid.uuid1()}.png"
    plt.savefig(plot_path)
    print(f"Figure saved as: {plot_path}")

    # Show image
    # plt.imshow()