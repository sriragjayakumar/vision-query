from urllib.parse import parse_qs, urlparse


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