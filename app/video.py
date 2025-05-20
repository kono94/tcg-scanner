from ultralytics import YOLO
import argparse
import threading
import queue
import time
import cv2
import json
from inferencer import infere
from pathlib import Path
from util import PROJECT_ROOT, find_image_path, set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video file with YOLO model")
    parser.add_argument("video", type=str, help="Video file name in resources/test_videos/ (e.g., IMG_3374.mp4)")

    # Parse arguments
    args = parser.parse_args()
    video_name = args.video
    
    print(video_name)
    model = YOLO(PROJECT_ROOT / 'app' / 'weights' / 'card_detector.pt', verbose=False)

    cap = cv2.VideoCapture(PROJECT_ROOT / 'resources' / 'test_videos' / video_name)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(PROJECT_ROOT / 'resources' / 'processed' / video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    tracks = {}
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    min_conf = 0.8  # Minimum confidence threshold for recognition
    final_max_conf = 0.98
    recognition_queue = queue.Queue()

    def process_recognition(track_id, cropped_image, min_conf):
        result, max_conf = infere(cropped_image, min_conf)
        if result is not None:
            card_id = Path(result).stem
            json_path = find_image_path(f"{card_id}.json")
            name = "Unknown"
            if json_path and json_path.exists():
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    name = data.get("name", "Unknown")

            display_text = f"{result}\n{name}"
            tracks[track_id]["display_text"] = display_text
            tracks[track_id]["recognition"] = result
            tracks[track_id]["max_conf"] = max_conf
            print(f"Track {track_id} recognized as: {result}")
            print(tracks[track_id])
        else:
            print(f"Track {track_id} recognition failed with confidence {max_conf}. Retrying...")
        time.sleep(0.5)
        tracks[track_id]["retry"] = True


    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model.track(frame, conf=0.4, persist=True, tracker=PROJECT_ROOT / 'app' / 'tracker_config.yaml' , verbose=False)
            
            for box in results[0].boxes:
                if not box.is_track:
                    continue
                
                track_id = int(box.id)  # Unique ID for the track
                b = box.xyxy[0].tolist()  # Bounding box coordinates
                
                cropped_image = frame[int(b[1]):int(b[3]), int(b[0]):int(b[2])]  # Crop the image

                if track_id not in tracks:
                    tracks[track_id] = {"recognition": None, "retry": True, "display_text": None, "max_conf": 0.0}

                if  tracks[track_id]["recognition"] is None or (tracks[track_id]["retry"] is True):
                    # Recognition has not been done or failed, so perform it asynchronously
                    tracks[track_id]["retry"] = False
                    recognition_thread = threading.Thread(target=process_recognition, args=(track_id, cropped_image, 0.6))
                    recognition_thread.start()
                    
                if tracks[track_id]["recognition"] is not None:
                    cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 5)
                    
                    display_text = tracks[track_id]["display_text"]
                    lines = display_text.split('\n')
                    max_text_width = max([cv2.getTextSize(line, font, font_scale, font_thickness)[0][0] for line in lines])
                    total_text_height = len(lines) * cv2.getTextSize(lines[0], font, font_scale, font_thickness)[0][1] + (len(lines) - 1) * 5

                    text_position = (int(b[0]), int(b[1]) - total_text_height - 15)

                    padding = 5
                    cv2.rectangle(frame, 
                                (text_position[0] - padding, text_position[1] - padding), 
                                (text_position[0] + max_text_width + padding, text_position[1] + total_text_height + padding), 
                                (255, 255, 255), 
                                thickness=cv2.FILLED)

                    y0 = text_position[1] + cv2.getTextSize(lines[0], font, font_scale, font_thickness)[0][1]
                    for i, line in enumerate(lines):
                        y = y0 + i * (cv2.getTextSize(line, font, font_scale, font_thickness)[0][1] + 5)
                        cv2.putText(frame, line, (text_position[0], y), font, font_scale, (0, 0, 0), font_thickness)
            cv2.imshow('Frame', frame)
            out.write(frame)
            # Break the loop if the 'q' key is pressed
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        else:
            break
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()