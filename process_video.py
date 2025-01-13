import cv2
from ultralytics import YOLO
import numpy as np
import argparse
import os

from utils.image_utils import get_roi, modify_frame
from gaussian_blur import apply_gaussian_blur_opencv


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process video with YOLO model.")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input video file"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to YOLO model file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="video_output",
        help="Output directory for processed videos",
    )
    return parser.parse_args()


def initialize_video_writer(output_path, fourcc, fps, frame_size):
    return cv2.VideoWriter(
        filename=output_path, fourcc=fourcc, fps=fps, frameSize=frame_size, isColor=True
    )


def get_file_name(file_path: str):
    file_name = os.path.basename(file_path)
    file_name_without_extension = os.path.splitext(file_name)[0]
    return file_name_without_extension


def get_model_name(model_path: str):
    model_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
    return model_name


def process_video(input_path: str, model_path: str, output_dir: str):
    if not os.path.exists(input_path):
        raise RuntimeError(f"Input path '{input_path}' doesn't exist!")
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model path '{model_path}' doesn't exist!")

    # Input file
    cap = cv2.VideoCapture(input_path)

    # Get parameters of input video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_fps = cap.get(cv2.CAP_PROP_FPS)
    tot_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a cv2 video writer
    fourcc_mp4 = cv2.VideoWriter_fourcc(*"mp4v")
    file_name = get_file_name(input_path)
    model_name = get_model_name(model_path)

    DETECTION_VIDEO = os.path.join(output_dir, f"{file_name}-{model_name}-detection.mp4")
    BLUR_VIDEO = os.path.join(output_dir, f"{file_name}-{model_name}-blur.mp4")
    output_detection = initialize_video_writer(
        output_path=DETECTION_VIDEO,
        fourcc=fourcc_mp4,
        fps=frame_fps,
        frame_size=(width, height),
    )
    output_blur = initialize_video_writer(
        output_path=BLUR_VIDEO,
        fourcc=fourcc_mp4,
        fps=frame_fps,
        frame_size=(width, height),
    )

    # Load a model
    model = YOLO(model_path)

    # Process all frames
    frame_count = 0
    while True:
        print(
            f"\rProcessing ... {frame_count}/{tot_frame_count} frame",
            end="",
            flush=True,
        )
        ret, frame = cap.read()
        if not ret:
            break
        # Copy frames for downstream processing
        frame_rect = frame.copy()
        frame_blur = frame.copy()

        # Perform detection on an image
        results = model.predict(source=frame, save=False, save_txt=False, verbose=False)
        for result in results:
            boxes = result.boxes.xyxy
            for box in np.array(boxes):
                pt1 = tuple(box[:2].astype(int))
                pt2 = tuple(box[2:].astype(int))

                frame_rect = cv2.rectangle(
                    img=frame_rect, pt1=pt1, pt2=pt2, color=(0, 0, 255), thickness=3
                )

                # Get the region of interest and apply blur
                roi = get_roi(frame_blur, pt1, pt2)
                blurred_roi = apply_gaussian_blur_opencv(roi)
                modify_frame(
                    frame=frame_blur, blurred_roi=blurred_roi, pt1=pt1, pt2=pt2
                )

        # Write to output
        output_detection.write(frame_rect)
        output_blur.write(frame_blur)
        # Increase frame count
        frame_count += 1

    # Display logs
    print()
    print(f"Output saved as: {DETECTION_VIDEO}")
    print(f"Output saved as: {BLUR_VIDEO}")

    # Close video files
    cap.release()
    output_detection.release()
    output_blur.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_arguments()
    process_video(args.input, args.model, args.output)
