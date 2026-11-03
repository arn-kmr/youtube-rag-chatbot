import os
import re
import tempfile
from typing import List
from dotenv import load_dotenv, find_dotenv
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from langchain_core.documents import Document
from src.utils import format_timestamp, parse_time_to_seconds

load_dotenv(find_dotenv())


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


def _transcribe_audio_with_gemini(video_id: str) -> List[dict]:
    """
    Fallback: Attempts to download audio with yt-dlp and transcribe with Gemini.
    Gracefully catches YouTube's bot-detection (HTTP 403 / SABR).
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
                    "player_client": ["android", "web"]
                }
            }
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        except Exception as yt_err:
            err_str = str(yt_err)
            if "403" in err_str or "Forbidden" in err_str or "Requested format is not available" in err_str:
                raise ValueError(
                    "This YouTube video has captions/transcripts disabled by the creator, "
                    "and YouTube's bot protection (HTTP 403 Forbidden) prevented downloading the audio stream. "
                    "Please test with a video that has subtitles or closed captions enabled (e.g. podcasts, tutorials, talks, or interviews)."
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
            model="gemini-2.5-flash",
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


def fetch_transcript_documents(
    video_id: str,
    chunk_size: int = 800,
) -> List[Document]:
    """
    Fetches the transcript for a YouTube video and chunks it with timestamps.
    Hierarchy:
    1. Direct transcript fetch via YouTube Transcript API.
    2. Auto-generated / translated transcript to English or native language.
    3. Multi-modal audio fallback using yt-dlp + Gemini if captions are disabled.
    """
    api = YouTubeTranscriptApi()

    # 1. Try direct fetch
    try:
        raw_snippets = api.fetch(video_id)
        if raw_snippets:
            return _chunk_snippets(raw_snippets, video_id, chunk_size)
    except Exception:
        pass

    # 2. Try multi-language / auto-generated / translation fallback
    try:
        transcript_list = api.list(video_id)
        found_transcript = None

        # Look for English (manual or auto)
        for t in transcript_list:
            if t.language_code.startswith("en"):
                found_transcript = t
                break

        # If no English transcript, try auto-translating the first available one to English
        if not found_transcript:
            for t in transcript_list:
                if t.is_translatable:
                    try:
                        found_transcript = t.translate("en")
                        break
                    except Exception:
                        continue

        # If translation failed, use first available native transcript directly
        if not found_transcript:
            found_transcript = next(iter(transcript_list), None)

        if found_transcript:
            raw_snippets = found_transcript.fetch()
            if raw_snippets:
                return _chunk_snippets(raw_snippets, video_id, chunk_size)
    except Exception:
        pass

    # 3. Third-tier fallback: Audio download + Gemini transcription
    try:
        raw_snippets = _transcribe_audio_with_gemini(video_id)
        if raw_snippets:
            return _chunk_snippets(raw_snippets, video_id, chunk_size)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            f"This video does not have captions or transcripts available. Details: {str(e)}"
        )

    raise ValueError(
        "No captions or subtitles were found for this video. Please select a video with captions enabled."
    )
