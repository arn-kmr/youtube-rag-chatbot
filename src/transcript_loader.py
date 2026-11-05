import os
import re
import json
import tempfile
import urllib.request
from typing import List, Optional
from dotenv import load_dotenv, find_dotenv
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from langchain_core.documents import Document
try:
    from src.utils import format_timestamp, parse_time_to_seconds
except ImportError:
    from utils import format_timestamp, parse_time_to_seconds

load_dotenv(find_dotenv())

SAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data")


def _chunk_snippets(snippets: list, video_id: str, chunk_size: int = 800) -> List[Document]:
    """Helper to merge raw snippets into timestamp-preserved Document chunks."""
    documents: List[Document] = []
    current_snippets = []
    current_char_count = 0

    for snippet in snippets:
        text = snippet.text.strip().replace("\n", " ") if hasattr(snippet, "text") else snippet.get("text", "").strip().replace("\n", " ")
        start = snippet.start if hasattr(snippet, "start") else snippet.get("start", 0.0)

        if not text:
            continue

        current_snippets.append({"text": text, "start": start})
        current_char_count += len(text) + 1

        if current_char_count >= chunk_size:
            start_time = current_snippets[0]["start"]
            timestamp_str = format_timestamp(start_time)
            chunk_text = " ".join(s["text"] for s in current_snippets)

            documents.append(
                Document(
                    page_content=f"[{timestamp_str}] {chunk_text}",
                    metadata={
                        "start_seconds": start_time,
                        "timestamp": timestamp_str,
                        "video_id": video_id,
                    },
                )
            )

            overlap_count = max(1, len(current_snippets) // 4)
            current_snippets = current_snippets[-overlap_count:]
            current_char_count = sum(len(s["text"]) + 1 for s in current_snippets)

    if current_snippets:
        start_time = current_snippets[0]["start"]
        timestamp_str = format_timestamp(start_time)
        chunk_text = " ".join(s["text"] for s in current_snippets)
        documents.append(
            Document(
                page_content=f"[{timestamp_str}] {chunk_text}",
                metadata={
                    "start_seconds": start_time,
                    "timestamp": timestamp_str,
                    "video_id": video_id,
                },
            )
        )

    return documents


def _extract_captions_with_ytdlp(video_id: str) -> List[dict]:
    """
    Extracts manual or auto-generated YouTube subtitles using yt-dlp's player clients
    (visionos, android, web) WITHOUT downloading any media stream.
    Bypasses YouTube's 403 Forbidden / SABR protection by streaming timedtext json3 directly.
    """
    import yt_dlp

    ydl_opts = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web", "visionos"]
            }
        },
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            if not info:
                return []

            subtitles = info.get("subtitles", {})
            auto_captions = info.get("automatic_captions", {})

            # Priority 1: English subtitles (manual or auto)
            target_track = None
            for key in ["en", "en-US", "en-GB", "en-CA", "en-IN"]:
                if key in subtitles and subtitles[key]:
                    target_track = subtitles[key]
                    break
                if key in auto_captions and auto_captions[key]:
                    target_track = auto_captions[key]
                    break

            # Priority 2: Any available subtitle track
            if not target_track:
                all_subs = {**subtitles, **auto_captions}
                for lang_code, formats in all_subs.items():
                    if formats:
                        target_track = formats
                        break

            if not target_track:
                return []

            # Find json3 format for structured timestamps
            json3_url = next((f["url"] for f in target_track if f.get("ext") == "json3"), None)
            if not json3_url:
                # Fallback to first available URL
                json3_url = target_track[0].get("url")

            if not json3_url:
                return []

            req = urllib.request.Request(
                json3_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            snippets = []
            for event in data.get("events", []):
                start = event.get("tStartMs", 0) / 1000.0
                segs = event.get("segs", [])
                text = "".join(s.get("utf8", "") for s in segs).strip()
                if text and text != "\n":
                    snippets.append({"text": text, "start": start})

            return snippets
    except Exception:
        return []


def _transcribe_audio_with_gemini(video_id: str) -> List[dict]:
    """
    Fallback: Attempts to download audio with yt-dlp and transcribe with Gemini.
    Used only when captions cannot be extracted via API or yt-dlp subtitle tracks.
    """
    import yt_dlp
    from google import genai

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "Transcripts are disabled for this video, and GOOGLE_API_KEY is required for audio transcription fallback."
        )

    client = genai.Client(api_key=api_key)

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_template = os.path.join(tmpdir, "audio.%(ext)s")
        ydl_opts = {
            "format": "ba[ext=m4a]/ba/b",
            "outtmpl": audio_template,
            "quiet": True,
            "no_warnings": True,
            "extractor_args": {
                "youtube": {
                    "player_client": ["android", "web", "visionos"]
                }
            },
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        except Exception as yt_err:
            err_str = str(yt_err)
            if "403" in err_str or "Forbidden" in err_str or "Requested format is not available" in err_str:
                raise ValueError(
                    "YouTube blocked media downloading (HTTP 403 Forbidden). "
                    "This video does not have available captions, and datacenter protection prevented audio fallback. "
                    "Please select a video with closed captions enabled, or paste the transcript directly."
                )
            raise ValueError(f"Audio fallback failed: {err_str}")

        downloaded_files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if not f.endswith(".part")]
        if not downloaded_files:
            raise ValueError("Failed to retrieve audio stream for transcription fallback.")

        audio_file_path = downloaded_files[0]
        uploaded_file = client.files.upload(file=audio_file_path)

        prompt = (
            "Transcribe this audio verbatim. Provide the transcript broken down with timestamps "
            "in the format: [MM:SS] spoken text. Include regular timestamp anchors."
        )

        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=[uploaded_file, prompt],
        )

        snippets = []
        lines = response.text.split("\n")
        ts_pattern = re.compile(r"\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.*)")
        for line in lines:
            m = ts_pattern.search(line)
            if m:
                ts_str, txt = m.group(1), m.group(2).strip()
                secs = parse_time_to_seconds(ts_str)
                snippets.append({"text": txt, "start": secs})
            elif line.strip() and snippets:
                snippets[-1]["text"] += " " + line.strip()

        return snippets


def create_documents_from_text(raw_text: str, video_id: str, chunk_size: int = 800) -> List[Document]:
    """
    Parses a manually pasted or uploaded transcript text and creates timestamp-aware Document chunks.
    Extracts [MM:SS] or [HH:MM:SS] timestamps if present, or creates linear timestamps.
    """
    lines = raw_text.splitlines()
    snippets = []
    ts_pattern = re.compile(r"\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?\s*(.*)")

    current_sec = 0.0
    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
        m = ts_pattern.match(line_str)
        if m:
            time_str, txt = m.group(1), m.group(2).strip()
            current_sec = float(parse_time_to_seconds(time_str))
            if txt:
                snippets.append({"text": txt, "start": current_sec})
        else:
            snippets.append({"text": line_str, "start": current_sec})
            current_sec += 5.0  # approximate linear offset if timestamps omitted

    if not snippets:
        snippets = [{"text": raw_text.strip(), "start": 0.0}]

    return _chunk_snippets(snippets, video_id, chunk_size=chunk_size)


def fetch_transcript_documents(
    video_id: str,
    chunk_size: int = 800,
) -> List[Document]:
    """
    Fetches the transcript for a YouTube video and chunks it with timestamps.
    Multi-tier resilient hierarchy:
    Tier 0: Pre-bundled local sample data (instant load, 0 network latency).
    Tier 1: Direct fetch via YouTube Transcript API.
    Tier 2: Multi-language / auto-generated / translated transcript.
    Tier 3: yt-dlp Subtitle Extraction (no audio download, bypasses 403 Forbidden).
    Tier 4: Multi-modal Gemini Audio fallback.
    """
    # 0. Check pre-bundled sample data (e.g. for demo video on cloud)
    sample_file = os.path.join(SAMPLE_DATA_DIR, f"{video_id}.json")
    if os.path.exists(sample_file):
        try:
            with open(sample_file, "r", encoding="utf-8") as f:
                raw_snippets = json.load(f)
            if raw_snippets:
                return _chunk_snippets(raw_snippets, video_id, chunk_size)
        except Exception:
            pass

    # 1. Direct fetch via YouTubeTranscriptApi
    try:
        api = YouTubeTranscriptApi()
        raw_snippets = api.fetch(video_id)
        if raw_snippets:
            return _chunk_snippets(raw_snippets, video_id, chunk_size)
    except Exception:
        pass

    # 2. Multi-language / translation fallback via YouTubeTranscriptApi
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        found_transcript = None

        for t in transcript_list:
            if t.language_code.startswith("en"):
                found_transcript = t
                break

        if not found_transcript:
            for t in transcript_list:
                if t.is_translatable:
                    try:
                        found_transcript = t.translate("en")
                        break
                    except Exception:
                        continue

        if not found_transcript:
            found_transcript = next(iter(transcript_list), None)

        if found_transcript:
            raw_snippets = found_transcript.fetch()
            if raw_snippets:
                return _chunk_snippets(raw_snippets, video_id, chunk_size)
    except Exception:
        pass

    # 3. yt-dlp subtitle stream extraction (bypasses YouTube datacenter block)
    try:
        ytdlp_snippets = _extract_captions_with_ytdlp(video_id)
        if ytdlp_snippets:
            return _chunk_snippets(ytdlp_snippets, video_id, chunk_size)
    except Exception:
        pass

    # 4. Gemini Audio Fallback
    try:
        raw_snippets = _transcribe_audio_with_gemini(video_id)
        if raw_snippets:
            return _chunk_snippets(raw_snippets, video_id, chunk_size)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            f"Unable to extract captions or audio for this video. Details: {str(e)}"
        )

    raise ValueError(
        "No captions or subtitles were found for this video. Please select a video with captions enabled, "
        "or paste the transcript manually in the 'Paste Transcript' section."
    )
