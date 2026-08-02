from urllib.parse import parse_qs, urlparse
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