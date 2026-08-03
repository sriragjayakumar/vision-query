import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional
import cv2
import streamlit as st
import torch
from ultralytics import YOLOWorld
from vidgear.gears import CamGear
from utils import clean_youtube_url, create_output_dir, make_zip, save_frame, timestamp_from_msec
st.set_page_config(page_title="Vision Query", layout="wide")

MODEL_PATH = "yolov8x-worldv2.pt"
OUTPUT_ROOT = Path("detectedImages")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = 0 if CUDA_AVAILABLE else "cpu"
USE_HALF = CUDA_AVAILABLE


@st.cache_resource()
def load_model(model_path: str, object_name: str) -> YOLOWorld:
    model = YOLOWorld(model_path)
    model.to("cpu")
    model.set_classes([object_name])
    return model


def process_video(
    source: str,
    object_name: str,
    confidence: float,
    frame_interval: int,
    image_size: int,
    jpeg_quality: int,
    max_images: int,
    is_online: bool,
) -> tuple[list[Path], dict]:
    output_dir = create_output_dir(OUTPUT_ROOT)
    # The cache key includes object_name, so a model already moved to CUDA by
    # an earlier Streamlit run is never reconfigured with CPU text tokens.
    model = load_model(MODEL_PATH, object_name)

    stream_options = {"source": source, "logging": False}
    if is_online:
        stream_options.update({"stream_mode": True, "STREAM_RESOLUTION": "360p"})

    stream = CamGear(**stream_options).start()
    progress = st.progress(0, text="Opening video...")
    live_status = st.empty()
    gallery_placeholder = st.container()

    saved_files: list[Path] = []
    frames_read = 0
    frames_processed = 0
    detections = 0
    started = time.perf_counter()

    # CamGear wraps a cv2.VideoCapture for normal files. Online sources may not expose a total.
    total_frames = 0
    fps = 0.0
    capture = getattr(stream, "stream", None)
    if capture is not None:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)

    try:
        with torch.inference_mode():
            while True:
                frame = stream.read()
                if frame is None:
                    break

                frames_read += 1
                if frames_read % frame_interval != 0:
                    continue

                frames_processed += 1
                result = model.predict(
                    source=frame,
                    conf=confidence,
                    imgsz=image_size,
                    device=DEVICE,
                    half=USE_HALF,
                    verbose=False,
                )[0]

                box_count = 0 if result.boxes is None else len(result.boxes)
                detections += box_count

                if capture is not None:
                    current_msec = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                else:
                    current_msec = 0.0
                timestamp = timestamp_from_msec(current_msec, frames_read, fps)

                if box_count > 0:
                    annotated = result.plot()
                    saved_path = save_frame(
                        annotated, output_dir, timestamp, frames_read, jpeg_quality
                    )
                    saved_files.append(saved_path)

                    with gallery_placeholder:
                        st.image(
                            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                            caption=f"{timestamp} · {box_count} detection(s)",
                            use_container_width=True,
                        )

                    if len(saved_files) >= max_images:
                        live_status.warning(f"Stopped after reaching the {max_images}-image limit.")
                        break

                elapsed = time.perf_counter() - started
                rate = frames_processed / elapsed if elapsed else 0.0
                live_status.info(
                    f"Read {frames_read:,} frames · Analysed {frames_processed:,} · "
                    f"Found {detections:,} objects · Saved {len(saved_files):,} images · {rate:.1f} FPS"
                )

                if total_frames > 0:
                    fraction = min(frames_read / total_frames, 1.0)
                    progress.progress(fraction, text=f"Processing {fraction:.0%}")
                else:
                    progress.progress(0, text=f"Processing frame {frames_read:,}...")
    finally:
        stream.stop()
        if CUDA_AVAILABLE:
            torch.cuda.synchronize()

    elapsed = time.perf_counter() - started
    progress.progress(1.0, text="Processing complete")
    zip_path = make_zip(saved_files, output_dir)

    return saved_files, {
        "frames_read": frames_read,
        "frames_processed": frames_processed,
        "detections": detections,
        "elapsed": elapsed,
        "output_dir": output_dir,
        "zip_path": zip_path,
    }


# st.title("Vision Query")
# st.caption("Upload a video or paste a video/stream URL, then extract frames containing the object you describe.")
st.markdown("""
<h1 style='text-align: center;'>
    Vision Query
</h1>
<p style='text-align: center; font-size:18px; color:gray;'>
    Upload a video or paste a video/stream URL, then extract frames containing the object you describe.
</p>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Detection settings")
    object_name = st.text_input("Object to detect", placeholder="e.g. person, red car, hard hat")
    confidence = st.slider("Confidence", 0.01, 1.00, 0.05, 0.01)
    frame_interval = st.number_input("Analyse every Nth frame", min_value=1, max_value=300, value=2)
    image_size = st.select_slider("Inference image size", options=[320, 480, 640, 960, 1280], value=640)
    jpeg_quality = st.slider("JPEG quality", 50, 100, 85)
    max_images = st.number_input("Maximum extracted images", min_value=1, max_value=1000, value=100)
    st.caption(f"Device: {'CUDA GPU' if CUDA_AVAILABLE else 'CPU'}")

source_type = st.radio("Video source", ["Upload video", "URL / stream"], horizontal=True)
source = ""
temp_dir: Optional[str] = None

if source_type == "Upload video":
    uploaded = st.file_uploader("Choose a video", type=["mp4", "mov", "avi", "mkv", "m4v", "webm"])
    if uploaded is not None:
        st.video(uploaded)
else:
    source = clean_youtube_url(st.text_input("Video, YouTube, or stream URL", placeholder="https://..."))

run = st.button("Extract detection images", type="primary", use_container_width=True)

if run:
    if not object_name.strip():
        st.error("Enter the object you want YOLO-World to detect.")
        st.stop()

    if source_type == "Upload video":
        if uploaded is None:
            st.error("Upload a video first.")
            st.stop()
        temp_dir = tempfile.mkdtemp(prefix="yolo_world_")
        suffix = Path(uploaded.name).suffix or ".mp4"
        local_path = Path(temp_dir) / f"input{suffix}"
        with local_path.open("wb") as handle:
            handle.write(uploaded.getbuffer())
        source = str(local_path)

    if not source:
        st.error("Enter a video URL or stream URL.")
        st.stop()

    try:
        files, stats = process_video(
            source=source,
            object_name=object_name.strip().lower(),
            confidence=float(confidence),
            frame_interval=int(frame_interval),
            image_size=int(image_size),
            jpeg_quality=int(jpeg_quality),
            max_images=int(max_images),
            is_online=source_type == "URL / stream",
        )

        st.success(f"Finished: extracted {len(files)} image(s).")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Frames read", f"{stats['frames_read']:,}")
        col2.metric("Frames analysed", f"{stats['frames_processed']:,}")
        col3.metric("Object detections", f"{stats['detections']:,}")
        col4.metric("Elapsed", f"{stats['elapsed']:.1f}s")

        if stats["zip_path"] is not None:
            with Path(stats["zip_path"]).open("rb") as archive:
                st.download_button(
                    "Download all extracted images (.zip)",
                    data=archive.read(),
                    file_name="yolo_world_detections.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
        else:
            st.info("No matching objects were detected. Try lowering confidence or analysing more frames.")
    except Exception as exc:
        st.exception(exc)
        st.info(
            "For partial-file MP4 errors, confirm the upload/download completed. "
            "For YouTube URLs, make sure yt-dlp and FFmpeg are installed."
        )
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
