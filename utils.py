import time
import zipfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, parse_qs
from yt_dlp import YoutubeDL
import cv2


def clean_youtube_url(url: str) -> str:
    """
    Convert YouTube URLs containing playlist, index, timestamp,
    and tracking parameters into a single-video URL.
    """
    parsed = urlparse(url.strip())
    hostname = parsed.hostname or ""
    hostname = hostname.lower().removeprefix("www.")
    if hostname in {"youtube.com", "m.youtube.com", "music.youtube.com"}:  # Standardize YouTube URL:
        query = parse_qs(parsed.query)
        video_id = query.get("v", [None])[0]
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"

    if hostname == "youtu.be": # Short YouTube URL:
        video_id = parsed.path.strip("/").split("/")[0]

        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
    return url.strip()


def timestamp_from_msec(milliseconds: float, frame_number: int, fps: float) -> str:
    if milliseconds <= 0 and fps > 0:
        milliseconds = (frame_number / fps) * 1000
    total_ms = max(0, int(milliseconds))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def create_output_dir(output_root: Path) -> Path:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_frame(frame, output_dir: Path, timestamp: str, frame_number: int, quality: int) -> Path:
    safe_timestamp = timestamp.replace(":", "-").replace(".", "-")
    path = output_dir / f"detection_{safe_timestamp}_frame_{frame_number:08d}.jpg"
    ok = cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"Could not save {path.name}")
    return path


def make_zip(files: list[Path], output_dir: Path) -> Optional[Path]:
    if not files:
        return None
    zip_path = output_dir / "detections.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, arcname=file.name)
    return zip_path


def get_video_fps(stream):
    try:
        video_fps = float(
            stream.stream.get(cv2.CAP_PROP_FPS)
        )
    except (AttributeError, TypeError, ValueError):
        video_fps = 0.0
    if video_fps <= 0 or video_fps > 240:
        video_fps = 30.0  #Fallback FPS
    return video_fps

def seconds_to_timestamp(total_seconds):
    """
    Convert seconds into HH:MM:SS.mmm.
    Example:
        28.367 -> 00:00:28.367
    """
    total_seconds = max(float(total_seconds), 0.0)
    total_milliseconds = round(total_seconds * 1000)
    hours = total_milliseconds // 3_600_000
    remaining = total_milliseconds % 3_600_000
    minutes = remaining // 60_000
    remaining %= 60_000
    seconds = remaining // 1000
    milliseconds = remaining % 1000
    return (f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}")


def get_video_timestamp(frame_number,stream):
    """
    Calculate the video timestamp from frame number and FPS.
    Frame 1 is treated as timestamp 00:00:00.000.
    """
    frame_index = max(frame_number - 1, 0)
    video_seconds = frame_index / get_video_fps(stream)
    return seconds_to_timestamp(video_seconds)





def get_playlist_urls(url: str):
    """
    Accepts either:
        https://www.youtube.com/watch?v=...&list=...
    or
        https://www.youtube.com/playlist?list=...

    Returns:
        list[str] of every video URL in the playlist.
    """

    parsed = urlparse(url)

    if "playlist" not in parsed.path:
        qs = parse_qs(parsed.query)

        if "list" not in qs:
            raise ValueError("This video is not part of a playlist.")

        playlist_id = qs["list"][0]
        url = f"https://www.youtube.com/playlist?list={playlist_id}"

    ydl_opts = {
        "quiet": True,
        "extract_flat": "in_playlist",
        "skip_download": True,
        "ignoreerrors": True,
        "playlistend": None,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    if "entries" not in info:
        raise RuntimeError("Could not extract playlist.")

    urls = []
    for entry in info["entries"]:
        if entry is None:
            continue

        video_id = entry.get("id")

        if video_id:
            urls.append(f"https://www.youtube.com/watch?v={video_id}")
    return urls
