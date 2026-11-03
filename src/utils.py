import re


def extract_video_id(url_or_id: str) -> str | None:
    """
    Extracts the 11-character YouTube video ID from various URL formats or raw ID.
    Supports:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://www.youtube.com/shorts/VIDEO_ID
    - VIDEO_ID (plain 11-character string)
    """
    if not url_or_id:
        return None
    url_or_id = url_or_id.strip()

    # Raw 11-character ID check
    if len(url_or_id) == 11 and re.match(r"^[a-zA-Z0-9_-]{11}$", url_or_id):
        return url_or_id

    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:[&?]|$)",
        r"(?:embed\/|v\/|shorts\/|youtu\.be\/)([0-9A-Za-z_-]{11})(?:[&?]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    return None


def format_timestamp(seconds: float) -> str:
    """Formats seconds into HH:MM:SS or MM:SS format."""
    total_secs = int(seconds)
    hours = total_secs // 3600
    minutes = (total_secs % 3600) // 60
    secs = total_secs % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def parse_time_to_seconds(time_str: str) -> int:
    """Parses MM:SS or HH:MM:SS into total integer seconds."""
    parts = list(map(int, time_str.split(":")))
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    elif len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0


def linkify_timestamps(text: str, video_id: str) -> str:
    """
    Finds timestamp citations like [04:25] or [01:15:30] in text and converts them
    into clickable Markdown links that jump directly to that point in the YouTube video.
    Avoids double-linking timestamps already inside a markdown link.
    """
    pattern = r"\[(\d{1,2}:\d{2}(?::\d{2})?)\](?!\()"

    def replace_ts(match):
        ts = match.group(1)
        secs = parse_time_to_seconds(ts)
        url = f"https://www.youtube.com/watch?v={video_id}&t={secs}s"
        return f"[[{ts}]]({url})"

    return re.sub(pattern, replace_ts, text)
