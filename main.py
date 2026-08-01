import cv2
import torch
from ultralytics import YOLO, YOLOWorld
from vidgear.gears import CamGear
import json

with open("pose_names.json", "r") as f:
    pose_map = json.load(f)

device = 0 if torch.cuda.is_available() else "cpu"
print(
    f"Using device: "
    f"{torch.cuda.get_device_name(0) if device == 0 else 'CPU'}"
)

# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------

VIDEO_SOURCE = input("Enter a YouTube URL, stream URL, or local video path: ").strip()
OBJECT_NAME = input("Enter the object you want to detect: ").strip().lower()

if not VIDEO_SOURCE:
    raise ValueError("A video source is required.")

if not OBJECT_NAME:
    raise ValueError("An object name is required.")


# ---------------------------------------------------------
# VIDEO SOURCE
# ---------------------------------------------------------

is_online_source = VIDEO_SOURCE.startswith(("http://", "https://"))
stream_options = {
    "source": VIDEO_SOURCE,
    "logging": True
}
if is_online_source:
    stream_options["stream_mode"] = True

stream = CamGear(**stream_options).start()


# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------

# YOLO-World detects the single object requested by the user.
world_model = YOLOWorld("yolov8x-worldv2.pt")
world_model.set_classes([OBJECT_NAME])

use_pose = any(OBJECT_NAME in aliases for aliases in pose_map.values())

pose_model = None

if use_pose:
    pose_model = YOLO("yolo26x-pose.pt")
    print("Pose estimation enabled for this body-related query.")
else:
    print("Pose estimation not required for this query.")


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

WORLD_CONFIDENCE = 0.25
POSE_CONFIDENCE = 0.35
IMAGE_SIZE = 640


# ---------------------------------------------------------
# VIDEO PROCESSING
# ---------------------------------------------------------

try:
    while True:
        frame = stream.read()

        if frame is None:
            break

        # Detect the object entered by the user.
        world_result = world_model.predict(
            source=frame,
            conf=WORLD_CONFIDENCE,
            imgsz=IMAGE_SIZE,
            device=device,
            verbose=False
        )[0]

        output_frame = frame.copy()

        # Draw only detections that passed the confidence threshold.
        if world_result.boxes is not None:
            for box in world_result.boxes:
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                confidence = float(box.conf[0])

                cv2.rectangle(
                    output_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                label = f"{OBJECT_NAME.upper()} {confidence:.2f}"

                cv2.putText(
                    output_frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA
                )

        # Run pose estimation only for body-related object names.
        if use_pose and pose_model is not None:
            pose_result = pose_model.predict(
                source=frame,
                conf=POSE_CONFIDENCE,
                imgsz=IMAGE_SIZE,
                device=device,
                verbose=False
            )[0]

            output_frame = pose_result.plot(
                img=output_frame,
                labels=False,
                boxes=False,
                conf=False,
                kpt_radius=3,
                kpt_line=True
            )

        cv2.imshow(
            f"Vision Query - Detecting: {OBJECT_NAME}",
            output_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    stream.stop()
    cv2.destroyAllWindows()