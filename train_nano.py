from ultralytics import YOLO
from pathlib import Path

from utils.device_utils import get_device


if __name__ == "__main__":
    # Load the model
    model_name = "yolo11n.pt"
    model = YOLO(model_name)

    # Train params
    BATCH = 64
    EPOCH = 50
    EXPT_NAME = model_name.replace(".pt", "") + "_" + f"b{BATCH}" + "_" + f"e{EPOCH}"

    # This is the path to my YAML relative to my CWD
    rel_path = "face_dataset/dataset.yaml"

    # This file _does_ exist
    assert Path(rel_path).exists(), "File doesn't exist"

    # Get device
    device = get_device()

    try:
        model.train(
            data=rel_path,
            epochs=EPOCH,
            imgsz=640,
            batch=BATCH,
            device=device,
            name=EXPT_NAME,
        )
    except RuntimeError as e:
        print(f"Runtime error: \n {e}")

        try:
            # Use full path
            full_path = Path(rel_path).resolve()
            model.train(
                data=str(full_path),
                epochs=EPOCH,
                imgsz=640,
                batch=BATCH,
                device=device,
                name=EXPT_NAME,
            )
        except RuntimeError as e:
            print(f"Runtime error: \n {e}")
