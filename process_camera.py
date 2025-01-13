from ultralytics import YOLO
import numpy as np
import cv2
import argparse

from gaussian_blur import apply_gaussian_blur, apply_gaussian_blur_opencv
from utils.device_utils import get_device
from utils.image_utils import get_roi, modify_frame

def load_model(model_path):
    return YOLO(model_path)

def initialize_camera(frame_width, frame_height):
    cap = cv2.VideoCapture(0)
    cap.set(propId=cv2.CAP_PROP_FRAME_WIDTH, value=frame_width)
    cap.set(propId=cv2.CAP_PROP_FRAME_HEIGHT, value=frame_height)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()
    return cap

def process_frame(frame, model):
    # Copy frames to avoid mutability
    frame_rect = frame.copy()
    frame_blur = frame.copy()

    results = model.predict(source=frame, save=False, save_txt=False)
    for result in results:
        boxes = result.boxes.xyxy
        for box in np.array(boxes):
            pt1 = tuple(box[:2].astype(int))
            pt2 = tuple(box[2:].astype(int))
            frame_rect = cv2.rectangle(
                img=frame_rect, pt1=pt1, pt2=pt2, color=(0, 0, 255), thickness=3
            )
            roi = get_roi(frame_blur, pt1, pt2)
            blurred_roi = apply_gaussian_blur_opencv(roi)
            modify_frame(frame=frame_blur, blurred_roi=blurred_roi, pt1=pt1, pt2=pt2)
    return frame_rect, frame_blur

def main():
    parser = argparse.ArgumentParser(description="Process camera input with YOLO model.")
    parser.add_argument("--model", required=True, help="Path to the YOLO model.")
    args = parser.parse_args()

    model_path = args.model
    frame_width = 1080
    frame_height = 720

    model = load_model(model_path)
    cap = initialize_camera(frame_width, frame_height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rect, frame_blur = process_frame(frame, model)
        cv2.imshow("FaceBlur", frame_blur)
        cv2.imshow("FaceDetection", frame_rect)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
