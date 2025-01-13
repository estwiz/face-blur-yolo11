# %%
import os
from ultralytics import YOLO
from utils.device_utils import get_device
import pandas as pd
import time
# %%

models = [
    "runs/detect/yolo11n_b64_e50/weights/best.pt",
    "runs/detect/yolo11s_b32_e50/weights/best.pt",
    "runs/detect/yolo11m_b16_e50/weights/best.pt"
]

number_of_frames = len(os.listdir("./face_dataset/images/val/"))

# Initialize an empty DataFrame to store the metrics
metrics_df = pd.DataFrame(
    columns=["model", "map50", "map75", "map_50_95", "mp", "mr", "f1", "fps"]
)

# Load the model
for model_path in models:
    model = YOLO(model_path)

    # Start the timer
    start_time = time.time()

    # Run the evaluation
    results = model.val(
        data="./face_dataset/dataset.yaml",
        device=get_device(),
        conf=0.001,
        iou=0.6,
        save=False,
        plots=False
    )

    # End the timer
    end_time = time.time()

    # Calculate the number of frames processed per second
    elapsed_time = end_time - start_time
    fps = number_of_frames / elapsed_time

    # Create a DataFrame for the current model's metrics
    model_metrics_df = pd.DataFrame([{
        "model": model_path,
        "map50": results.box.map50,
        "map75": results.box.map75,
        "map_50_95": results.box.maps[0],
        "mp": results.box.mp,
        "mr": results.box.mr,
        "f1": results.box.f1[0],
        "fps": fps,
    }])

    # Concatenate the current model's metrics with the main DataFrame
    metrics_df = pd.concat([metrics_df, model_metrics_df], ignore_index=True)

# Save the DataFrame to a CSV file
metrics_df.to_csv("model_evaluation_metrics.csv", index=False)
