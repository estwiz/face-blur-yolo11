# Use guide

## Create environment
This project requires `Python 3.10` or above.
1. Create a virtual environment
```shell
python3 -m venv .venv
```
2. Activate the environment
```shell
source ./.venv/bin/activate
```
3. Install packages
```shell
pip install -r requirements.txt
```

## Download dataset
1. Run the following command to download the dataset.
```shell
python3 download_dataset.py
```
2. You can see the visualization of sample images from the dataset by running the following command
```shell
python3 plot_dataset.py
```

## Train model
Run the following commands to train each model
```shell
# Nano model
python3 train_nano.py

# Small model
python3 train_small.py

# Medium model
python3 train_medium.py
```

## Evaluate model
Run the following command to evaluate the model
```shell
python3 model_evaluation.py
```

## Run inference on examples
### Video file
General command:
```shell
python3 process_video.py --input <path/to/input/file> --model <path/to/model> --output <path/to/output/folder>
```

Example usages:
```shell
# Yolov11 nano model
python3 process_video.py --input video_input/warsaw.mp4 --model runs/detect/yolo11n_b64_e50/weights/bests.pt

# Yolov11 small model
python3 process_video.py --input video_input/warsaw.mp4 --model runs/detect/yolo11s_b32_e50/weights/best.pt

# Yolov11 medium model
python3 process_video.py --input video_input/warsaw.mp4 --model runs/detect/yolo11m_b16_e50/weights/best.pt
```

### Camera feed
General command:
```shell
python3 process_camera.py --model <path/to/model>
```

Example usages:
```shell
# Yolov11 nano model
python3 process_camera.py --model runs/detect/yolo11n_b64_e50/weights/best.pt

# Yolov11 small model
python3 process_camera.py --model runs/detect/yolo11s_b32_e50/weights/best.pt

# Yolov11 medium model
python3 process_camera.py --model runs/detect/yolo11m_b16_e50/weights/best.pt
```