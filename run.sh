python3 process_video.py --input video_input/warsaw.mp4 --model runs/detect/yolo11n_b64_e50/weights/best.pt >> yolo11n.log &
python3 process_video.py --input video_input/warsaw.mp4 --model runs/detect/yolo11s_b32_e50/weights/best.pt >> yolo11s.log &
python3 process_video.py --input video_input/warsaw.mp4 --model runs/detect/yolo11m_b16_e50/weights/best.pt >> yolo11m.log &
wait
echo "Finished processing for video_input/warsaw.mp4"

python3 process_video.py --input video_input/new_york_1.mp4 --model runs/detect/yolo11n_b64_e50/weights/best.pt >> yolo11n.log &
python3 process_video.py --input video_input/new_york_1.mp4 --model runs/detect/yolo11s_b32_e50/weights/best.pt >> yolo11s.log &
python3 process_video.py --input video_input/new_york_1.mp4 --model runs/detect/yolo11m_b16_e50/weights/best.pt >> yolo11m.log &
wait
echo "Finished processing for video_input/new_york_1.mp4"

python3 process_video.py --input video_input/new_york_2.mp4 --model runs/detect/yolo11n_b64_e50/weights/best.pt >> yolo11n.log &
python3 process_video.py --input video_input/new_york_2.mp4 --model runs/detect/yolo11s_b32_e50/weights/best.pt >> yolo11s.log &
python3 process_video.py --input video_input/new_york_2.mp4 --model runs/detect/yolo11m_b16_e50/weights/best.pt >> yolo11m.log &
wait
echo "Finished processing for video_input/new_york_2.mp4"

python3 process_video.py --input video_input/new_york_3.mp4 --model runs/detect/yolo11n_b64_e50/weights/best.pt >> yolo11n.log &
python3 process_video.py --input video_input/new_york_3.mp4 --model runs/detect/yolo11s_b32_e50/weights/best.pt >> yolo11s.log &
python3 process_video.py --input video_input/new_york_3.mp4 --model runs/detect/yolo11m_b16_e50/weights/best.pt >> yolo11m.log &
wait
echo "Finished processing for video_input/new_york_3.mp4"


python3 process_video.py --input video_input/people_walking.mp4 --model runs/detect/yolo11n_b64_e50/weights/best.pt >> yolo11n.log &
python3 process_video.py --input video_input/people_walking.mp4 --model runs/detect/yolo11s_b32_e50/weights/best.pt >> yolo11s.log &
python3 process_video.py --input video_input/people_walking.mp4 --model runs/detect/yolo11m_b16_e50/weights/best.pt >> yolo11m.log &
wait
echo "Finished processing for video_input/people_walking.mp4"

python3 process_video.py --input video_input/golden_globes.mp4 --model runs/detect/yolo11n_b64_e50/weights/best.pt >> yolo11n.log &
python3 process_video.py --input video_input/golden_globes.mp4 --model runs/detect/yolo11s_b32_e50/weights/best.pt >> yolo11s.log &
python3 process_video.py --input video_input/golden_globes.mp4 --model runs/detect/yolo11m_b16_e50/weights/best.pt >> yolo11m.log &
wait
echo "Finished processing for video_input/golden_globes.mp4"