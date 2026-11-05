# 🎥 YouTube RAG Chatbot

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://youtube-rag-chatbot123.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-arn--kmr%2Fyoutube--rag--chatbot-blue?logo=github)](https://github.com/arn-kmr/youtube-rag-chatbot)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> 🚀 **Live Demo:** Try the application right now in your browser:  
> **👉 [https://youtube-rag-chatbot123.streamlit.app/](https://youtube-rag-chatbot123.streamlit.app/)**

---

An intelligent, interactive Retrieval-Augmented Generation (RAG) assistant that allows users to chat with any YouTube video. The assistant grounds answers strictly in the transcript, provides **clickable timestamp jump links** (`[04:12]`), offers **in-app video player seeking**, utilizes **Hybrid Retrieval (Dense FAISS + BM25)**, and maintains conversational memory.

---

## 🌟 Key Features

- **🌐 Live Cloud Demo**: Accessible worldwide at [youtube-rag-chatbot123.streamlit.app](https://youtube-rag-chatbot123.streamlit.app/).
- **Any YouTube Video**: Paste any standard (`watch?v=`), short (`/shorts/`), or embed URL.
- **Clickable Timestamp Links & In-App Seek**: Cites moments like `[04:25]`; click external links or use the `▶️ Seek Player to [MM:SS]` button to jump the embedded video player straight to that moment.
- **Rich Video Info Card**: Automatically retrieves video title, creator name, and indexed segment counts via YouTube's public oEmbed API.
- **Hybrid Retrieval (Dense + BM25)**: Combines semantic vector similarity (`all-MiniLM-L6-v2`) with keyword search (`BM25`) via `EnsembleRetriever` for superior precision on technical terms and proper nouns.
- **Multi-Tier Resilient Caption Fallback**:
  1. *Pre-bundled sample cache* for instant, zero-latency demo video loading.
  2. *Direct YouTube Transcript API* fetch.
  3. *Multi-language and auto-generated captions* auto-translated to English.
  4. *Direct `yt-dlp` subtitle stream extraction* using `visionos`/`android` player clients to bypass YouTube's datacenter bot blocks (HTTP 403 Forbidden).
  5. *Gemini multimodal audio fallback* if closed captions are unavailable.
  6. *Custom / Pasted Transcript mode* to ingest lecture notes or manual transcripts directly.
- **FAISS Disk Caching**: Transcripts and vector indexes are cached so returning to a video loads in milliseconds.
- **Conversational Memory**: Multi-turn dialogue awareness for asking contextual follow-up questions.
- **Real-Time Streaming**: Token-by-token streaming output directly inside Streamlit chat bubbles.
- **Export Notes**: One-click **Download Chat (Markdown)** button to save all Q&A notes and timestamp citations.

---

## 🏗️ Architecture

```
YouTube URL / Custom Transcript
       │
       ▼
[1. Resilient Caption Ingestion & Timestamp Chunking] 
(Local Cache ➔ YouTube Transcript API ➔ yt-dlp Subtitles ➔ Gemini Audio Fallback)
       │
       ▼
[2. Embeddings & Vector Index Cache] 
(sentence-transformers/all-MiniLM-L6-v2 + FAISS Local Storage)
       │
       ▼
[3. Hybrid Retrieval] 
(FAISS Dense Vector Search 60% + BM25 Sparse Keyword Search 40%)
       │
       ▼
[4. Augmented Generation Chain] 
(Gemini 3.5/3.8 Flash + Multi-Turn Chat History + Strict Timestamp Citations)
       │
       ▼
[5. Interactive Streamlit Interface] 
(Live Embedded Video Player + Direct Seeking + Clickable YouTube Jump Links)
```

---

## 🚀 Local Setup

### 1. Prerequisites
- Python 3.10+
- Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/arn-kmr/youtube-rag-chatbot.git
cd youtube-rag-chatbot

# Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 4. Run Locally
```bash
streamlit run app.py
```

---

## 🌐 Cloud Deployment (Streamlit Community Cloud)

This application is deployed and hosted on **Streamlit Community Cloud**:

- **Repository**: [https://github.com/arn-kmr/youtube-rag-chatbot](https://github.com/arn-kmr/youtube-rag-chatbot)
- **Live URL**: [https://youtube-rag-chatbot123.streamlit.app/](https://youtube-rag-chatbot123.streamlit.app/)

### Secrets Configuration:
In the Streamlit Cloud dashboard under **Settings** $\to$ **Secrets**:
```toml
GOOGLE_API_KEY = "your_actual_gemini_api_key"
```
