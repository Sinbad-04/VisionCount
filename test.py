from ultralytics import YOLO
import cv2
import torch
from ultralytics import solutions
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model = "model.pt"

video_path = "vd35.mp4"

cap = cv2.VideoCapture(video_path)


def count_objects_in_region(video_path, output_video_path, model_path, classes_to_count):
    """Count objects in a specific region within a video."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    region_points = [(477, 0), (803, 0), (803, 720), (477, 720)]
    counter = solutions.ObjectCounter(show=False, region=region_points, model=model_path, classes=classes_to_count)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or processing is complete.")
            break
        overlay = im0.copy()
        cv2.fillPoly(overlay, [np.array(region_points, dtype=np.int32)], (0, 255, 0))
        im0 = cv2.addWeighted(overlay, 0.3, im0, 0.7, 0)
        yl_result = counter.model(im0)[0]
        results = counter(im0)
        box_count = 0
        def_box_count = 0
        for box in yl_result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            score = float(box.conf[0])
            cls = int((box.cls[0].item()))
            cls_name = counter.model.names[cls]


            if cls == 0:
                box_count = counter.in_count
            else:
                def_box_count = counter.in_count

            text = f"Box: {box_count}"
            cv2.putText(im0, text, (850, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
            text_df = f"Defect Box: {def_box_count}"
            cv2.putText(im0, text_df, (850, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
        video_writer.write(results.plot_im)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


count_objects_in_region(video_path=video_path, output_video_path="output_video.mp4", model_path=model, classes_to_count=[0, 1])




