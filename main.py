import time
from pathlib import Path
import cv2
import torch
from ultralytics import YOLOWorld
from vidgear.gears import CamGear
from utils import clean_youtube_url, get_video_timestamp

# =========================================================
# CONFIGURATION
# =========================================================

WORLD_MODEL_PATH = "yolov8x-worldv2.pt"
WORLD_CONFIDENCE = 0.05
IMAGE_SIZE = 640
FRAME_INTERVAL = 1
BATCH_SIZE = 4
SAVE_DETECTIONS = True
JPEG_QUALITY = 85
OUTPUT_DIRECTORY = Path("detectedImages")
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

# =========================================================
# DEVICE
# =========================================================

cuda_available = torch.cuda.is_available()
device = 0 if cuda_available else "cpu"
use_half = cuda_available

if cuda_available:
    print(f"Using device: {torch.cuda.get_device_name(0)}")
else:
    print("Using device: CPU")

def synchronize_gpu():
    if cuda_available:
        torch.cuda.synchronize()

# =========================================================
# USER INPUT
# =========================================================
video_source = input("Enter a YouTube URL, stream URL, or local video path: ").strip()
video_source = clean_youtube_url(video_source)
object_name = input("Enter the object you want to detect: ").strip().lower()

if not video_source:
    raise ValueError("A video source is required.")

if not object_name:
    raise ValueError("An object name is required.")

# =========================================================
# VIDEO SOURCE
# =========================================================

is_online_source = video_source.startswith(("http://", "https://"))
stream_options = {
    "source": video_source,
    "logging": True,
    "STREAM_RESOLUTION": "360p"
}
if is_online_source:
    stream_options["stream_mode"] = True

stream = CamGear(**stream_options).start()
# =========================================================
# LOAD MODELS
# =========================================================

print("Loading YOLO-World model...")
world_model = YOLOWorld(WORLD_MODEL_PATH)
world_model.set_classes([object_name])

# =========================================================
# SAVE DETECTION IMAGE
# =========================================================

def save_detection_frame(output_frame,frame_number,timestamp):
    """
    Save every detected frame using its video timestamp.
    Example filename:
        detection_00-00-28-367.jpg
    """
    if not SAVE_DETECTIONS:
        return False

    safe_timestamp = timestamp.replace(":", "-").replace(".", "-")

    output_path = (
        OUTPUT_DIRECTORY
        / f"detection_{safe_timestamp}.jpg"
    )

    if output_path.exists():
        output_path = (
            OUTPUT_DIRECTORY
            / (
                f"detection_{safe_timestamp}_"
                f"frame_{frame_number:08d}.jpg"
            )
        )

    saved = cv2.imwrite(
        str(output_path),
        output_frame,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )

    if saved:
        print(
            f"Saved detection at {timestamp}: "
            f"{output_path}"
        )
    else:
        print(
            f"Failed to save detection at {timestamp}: "
            f"{output_path}"
        )

    return saved


# =========================================================
# PROCESS BATCH
# =========================================================
def draw_world_detections(frame, result, timestamp):
    """
    Return the frame and the actual number of YOLO-World detections.
    """
    output_frame = frame.copy()

    if result.boxes is None or len(result.boxes) == 0:
        return output_frame, 0

    detection_count = len(result.boxes)
    return output_frame, detection_count


def process_batch(frames, frame_metadata):
    """
    Run YOLO-World detection on one batch.
    """
    if not frames:
        return {
            "processed": 0,
            "detected_frames": 0,
            "detections": 0,
            "saved": 0,
            "inference_seconds": 0.0
        }

    synchronize_gpu()
    inference_start = time.perf_counter()

    world_results = world_model.predict(
        source=frames,
        conf=WORLD_CONFIDENCE,
        imgsz=IMAGE_SIZE,
        device=device,
        half=use_half,
        verbose=False
    )


    synchronize_gpu()

    inference_seconds = (
        time.perf_counter() - inference_start
    )

    batch_detected_frames = 0
    batch_detection_count = 0
    batch_saved_count = 0

    for index, world_result in enumerate(world_results):
        frame = frames[index]
        metadata = frame_metadata[index]

        output_frame, detection_count = (
            draw_world_detections(
                frame=frame,
                result=world_result,
                timestamp=metadata["timestamp"]
            )
        )

        batch_detection_count += detection_count

        if detection_count > 0:
            batch_detected_frames += 1

            was_saved = save_detection_frame(
                output_frame=output_frame,
                frame_number=metadata["frame_number"],
                timestamp=metadata["timestamp"]
            )

            if was_saved:
                batch_saved_count += 1

    processed_count = len(frames)

    inference_fps = (
        processed_count / inference_seconds
        if inference_seconds > 0
        else 0.0
    )

    return {
        "processed": processed_count,
        "detected_frames": batch_detected_frames,
        "detections": batch_detection_count,
        "saved": batch_saved_count,
        "inference_seconds": inference_seconds
    }


    synchronize_gpu()

    print("Warm-up complete.")


# =========================================================
# VIDEO PROCESSING
# =========================================================

total_frames_read = 0
total_frames_processed = 0
total_detected_frames = 0
total_detections = 0
total_images_saved = 0
total_inference_seconds = 0.0

report_processed_frames = 0
report_start_time = time.perf_counter()

batch_frames = []
batch_metadata = []

start_code_time = time.perf_counter()

try:
    with torch.inference_mode():
        while True:
            frame = stream.read()
            if frame is None:
                break
            total_frames_read += 1
            if total_frames_read % FRAME_INTERVAL != 0:
                continue

            frame_timestamp = get_video_timestamp(total_frames_read, stream)
            batch_frames.append(frame)
            batch_metadata.append({
                "frame_number": total_frames_read,
                "timestamp": frame_timestamp
            })
            if len(batch_frames) < BATCH_SIZE:
                continue

            statistics = process_batch(frames=batch_frames,frame_metadata=batch_metadata)
            total_frames_processed += statistics["processed"]
            total_detected_frames += statistics["detected_frames"]
            total_detections += statistics["detections"]
            total_images_saved += statistics["saved"]
            total_inference_seconds += statistics["inference_seconds"]
            report_processed_frames += statistics["processed"]

            batch_frames.clear()
            batch_metadata.clear()

            report_elapsed = (time.perf_counter() - report_start_time)

            if report_elapsed >= 2.0:
                end_to_end_fps = (report_processed_frames / report_elapsed)


finally:
    synchronize_gpu()
    total_elapsed = (time.perf_counter() - start_code_time)

    overall_analysed_fps = (
        total_frames_processed / total_elapsed
        if total_elapsed > 0
        else 0.0
    )

    pure_inference_fps = (
        total_frames_processed
        / total_inference_seconds
        if total_inference_seconds > 0
        else 0.0
    )

    print("\n" + "=" * 55)
    print("PROCESSING COMPLETE")
    print("=" * 55)
    print(f"Total time: {total_elapsed:.3f} seconds")
    print(f"Total object detections:{total_detections}")
    print(f"Detection images saved: {total_images_saved}")
    print("=" * 55)
    stream.stop()
    cv2.destroyAllWindows()