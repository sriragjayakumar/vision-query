import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLOE
from vidgear.gears import CamGear

from utils import clean_youtube_url, get_video_timestamp


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = "yoloe-26l-seg.pt"

# Start at 0.05 while testing. Increase it if there are false detections.
MODEL_CONFIDENCE = 0.05

IMAGE_SIZE = 640

# 1 = analyse every frame
# 2 = analyse every second frame
# 3 = analyse every third frame
FRAME_INTERVAL = 1

BATCH_SIZE = 4

SAVE_DETECTIONS = True
JPEG_QUALITY = 85

STREAM_RESOLUTION = "360p"

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
    """
    Wait for all queued CUDA operations to finish.

    This makes the inference timing accurate.
    """
    if cuda_available:
        torch.cuda.synchronize()


# =========================================================
# USER INPUT
# =========================================================

video_source = input(
    "Enter a YouTube URL, stream URL, or local video path: "
).strip()

video_source = clean_youtube_url(video_source)

object_name = input(
    "Enter the object you want to detect: "
).strip().lower()

if not video_source:
    raise ValueError("A video source is required.")

if not object_name:
    raise ValueError("An object name is required.")


# =========================================================
# VIDEO SOURCE
# =========================================================

is_online_source = video_source.startswith(
    ("http://", "https://")
)

stream_options = {
    "source": video_source,
    "logging": True
}

if is_online_source:
    stream_options["stream_mode"] = True
    stream_options["STREAM_RESOLUTION"] = STREAM_RESOLUTION

stream = CamGear(**stream_options).start()


# =========================================================
# LOAD YOLOE MODEL
# =========================================================

print("Loading YOLOE model...")

model = YOLOE(MODEL_PATH)

# Set the text prompt once before processing the video.
model.set_classes([object_name])

print(f"YOLOE prompt set to: {object_name}")


# =========================================================
# SAVE DETECTION IMAGE
# =========================================================

def save_detection_frame(
    output_frame,
    frame_number,
    timestamp
):
    """
    Save a detected frame using its video timestamp.

    Example:
        detection_00-00-28-367.jpg
    """
    if not SAVE_DETECTIONS:
        return False

    safe_timestamp = (
        timestamp
        .replace(":", "-")
        .replace(".", "-")
    )

    output_path = (
        OUTPUT_DIRECTORY
        / f"detection_{safe_timestamp}.jpg"
    )

    # Avoid overwriting an existing image with the same timestamp.
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
# DRAW DETECTIONS
# =========================================================

def draw_detections(
    frame,
    result,
    timestamp
):
    """
    Draw YOLOE boxes, labels, confidence scores and masks.

    Return:
        annotated frame
        number of detections
    """
    if result.boxes is None or len(result.boxes) == 0:
        return frame.copy(), 0

    detection_count = len(result.boxes)

    # YOLOE segmentation results may contain both boxes and masks.
    # result.plot() draws all available annotations.
    output_frame = result.plot(
        img=frame.copy()
    )

    print(
        f"[{timestamp}] "
        f"{object_name}: "
        f"{detection_count} detection(s)"
    )

    return output_frame, detection_count


# =========================================================
# PROCESS BATCH
# =========================================================

def process_batch(
    frames,
    frame_metadata
):
    """
    Run YOLOE inference on one batch of frames.
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

    results = model.predict(
        source=frames,
        conf=MODEL_CONFIDENCE,
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

    for index, result in enumerate(results):
        frame = frames[index]
        metadata = frame_metadata[index]

        output_frame, detection_count = draw_detections(
            frame=frame,
            result=result,
            timestamp=metadata["timestamp"]
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

    print(
        f"Batch size: {processed_count} | "
        f"Inference: {inference_seconds * 1000:.1f} ms | "
        f"Throughput: {inference_fps:.2f} FPS"
    )

    return {
        "processed": processed_count,
        "detected_frames": batch_detected_frames,
        "detections": batch_detection_count,
        "saved": batch_saved_count,
        "inference_seconds": inference_seconds
    }


# =========================================================
# UPDATE TOTAL STATISTICS
# =========================================================

def update_statistics(
    statistics,
    totals
):
    """
    Add one batch's statistics to the overall totals.
    """
    totals["processed"] += statistics["processed"]
    totals["detected_frames"] += statistics["detected_frames"]
    totals["detections"] += statistics["detections"]
    totals["saved"] += statistics["saved"]
    totals["inference_seconds"] += statistics["inference_seconds"]


# =========================================================
# VIDEO PROCESSING
# =========================================================

totals = {
    "frames_read": 0,
    "processed": 0,
    "detected_frames": 0,
    "detections": 0,
    "saved": 0,
    "inference_seconds": 0.0
}

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

            totals["frames_read"] += 1

            # Skip frames before inference.
            if totals["frames_read"] % FRAME_INTERVAL != 0:
                continue

            frame_timestamp = get_video_timestamp(
                totals["frames_read"],
                stream
            )

            batch_frames.append(frame)

            batch_metadata.append({
                "frame_number": totals["frames_read"],
                "timestamp": frame_timestamp
            })

            # Wait until the batch is full.
            if len(batch_frames) < BATCH_SIZE:
                continue

            statistics = process_batch(
                frames=batch_frames,
                frame_metadata=batch_metadata
            )

            update_statistics(
                statistics=statistics,
                totals=totals
            )

            report_processed_frames += statistics["processed"]

            batch_frames.clear()
            batch_metadata.clear()

            report_elapsed = (
                time.perf_counter() - report_start_time
            )

            if report_elapsed >= 2.0:
                end_to_end_fps = (
                    report_processed_frames / report_elapsed
                )

                print(
                    f"End-to-end analysed FPS: "
                    f"{end_to_end_fps:.2f} | "
                    f"Frames read: {totals['frames_read']} | "
                    f"Frames analysed: {totals['processed']}"
                )

                report_start_time = time.perf_counter()
                report_processed_frames = 0

        # Process frames left over after the final full batch.
        if batch_frames:
            statistics = process_batch(
                frames=batch_frames,
                frame_metadata=batch_metadata
            )

            update_statistics(
                statistics=statistics,
                totals=totals
            )


finally:
    synchronize_gpu()

    total_elapsed = (
        time.perf_counter() - start_code_time
    )

    overall_analysed_fps = (
        totals["processed"] / total_elapsed
        if total_elapsed > 0
        else 0.0
    )

    pure_inference_fps = (
        totals["processed"] / totals["inference_seconds"]
        if totals["inference_seconds"] > 0
        else 0.0
    )

    frames_skipped = (
        totals["frames_read"] - totals["processed"]
    )

    print("\n" + "=" * 55)
    print("PROCESSING COMPLETE")
    print("=" * 55)

    print(
        f"Total time: "
        f"{total_elapsed:.3f} seconds"
    )

    print(
        f"Frames read: "
        f"{totals['frames_read']}"
    )

    print(
        f"Frames analysed: "
        f"{totals['processed']}"
    )

    print(
        f"Frames skipped: "
        f"{frames_skipped}"
    )

    print(
        f"Frames containing detections: "
        f"{totals['detected_frames']}"
    )

    print(
        f"Total object detections: "
        f"{totals['detections']}"
    )

    print(
        f"Detection images saved: "
        f"{totals['saved']}"
    )

    print(
        f"Overall analysed FPS: "
        f"{overall_analysed_fps:.2f}"
    )

    print(
        f"Pure inference FPS: "
        f"{pure_inference_fps:.2f}"
    )

    print("=" * 55)

    stream.stop()
    cv2.destroyAllWindows()