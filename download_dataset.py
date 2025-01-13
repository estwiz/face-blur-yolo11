"""
Adapted from https://docs.voxel51.com/tutorials/yolov8.html#Curate-data-for-fine-tuning
"""

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.random as four


def export_yolo_data(
    samples, export_dir, classes, label_field="ground_truth", split=None
):

    if isinstance(split, list):
        splits = split
        for split in splits:
            export_yolo_data(samples, export_dir, classes, label_field, split)
    else:
        if split is None:
            split_view = samples
            split = "val"
        else:
            split_view = samples.match_tags(split)

        split_view.export(
            export_dir=export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field=label_field,
            classes=classes,
            split=split,
        )


# Create dataset
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    classes=["Human face"],
    max_samples=10_000,
    only_matching=True,
    label_types="detections",
    dataset_name="human-face-dataset",
    persistent=True,
).map_labels(field="ground_truth", map={"Human face": "human-face"})
dataset.save()

# Delete existing tags
dataset.untag_samples(dataset.distinct("tags"))

# Split into train and val dataset
four.random_split(dataset, {"train": 0.8, "val": 0.2})

# Export data in YOLO format
classes = ["human-face"]
export_yolo_data(
    samples=dataset, export_dir="face_dataset", classes=classes, split=["train", "val"]
)

# session = fo.launch_app(dataset)
# session.wait(-1)
