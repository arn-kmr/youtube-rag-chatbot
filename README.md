# 🎥 YouTube RAG Chatbot

An intelligent, interactive Retrieval-Augmented Generation (RAG) assistant that allows users to chat with any YouTube video. The assistant grounds answers strictly in the transcript, provides **clickable timestamp jump links** (`[04:12]`), utilizes **Hybrid Retrieval (Dense FAISS + BM25)**, and maintains conversational memory.

---

## 🌟 Key Features

- **Any YouTube Video**: Paste any standard (`watch?v=`), short (`/shorts/`), or embed URL.
- **Clickable Timestamp Links**: Cites moments like `[04:25]`; clicking jumps the video player straight to that second.
- **Hybrid Retrieval (Dense + BM25)**: Combines semantic vector similarity (`all-MiniLM-L6-v2`) with keyword search (`BM25`) via `EnsembleRetriever` for superior precision on technical jargon, proper nouns, and acronyms.
- **Multi-Tier Transcript Fallback**:
  1. Direct transcript fetch via YouTube Transcript API.
  2. Multi-language and auto-generated captions auto-translated to English.
  3. Audio download (`yt-dlp`) + Gemini audio transcription fallback when captions are disabled.
- **FAISS Disk Caching**: Transcripts and vector indexes are cached in `.faiss_cache/` so returning to a video loads instantly.
- **Conversational Memory**: Multi-turn dialogue awareness for asking follow-ups.
- **Real-Time Streaming**: Token-by-token streaming output directly inside Streamlit chat bubbles.
- **Side-by-Side Video Player**: Watch the video and jump to timestamps while conversing with the assistant.

---

## 🏗️ Architecture

```
YouTube URL / ID
       │
       ▼
[1. Transcript Extraction & Chunking with Timestamps] (youtube-transcript-api / Gemini Audio Fallback)
       │
       ▼
[2. Embeddings & Vector Index Cache] (sentence-transformers/all-MiniLM-L6-v2 + FAISS)
       │
       ▼
[3. Hybrid Retrieval] (FAISS Dense Vector Search 60% + BM25 Sparse Search 40%)
       │
       ▼
[4. Augmented Generation Chain] (Gemini 3.5/3.8 Flash + Chat History + Timestamp Anchors)
       │
       ▼
[5. Interactive Streamlit UI] (Interactive Video Player + Clickable Jump Links)
```

---

## 🚀 Local Setup

### 1. Prerequisites
- Python 3.10+
- Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/youtube-rag-chatbot.git
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

## 🌐 Free Cloud Deployment (Streamlit Community Cloud)

You can deploy this application for free on **Streamlit Community Cloud**:

1. **Initialize Git & Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "feat: complete YouTube RAG chatbot with hybrid search and clickable timestamps"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/youtube-rag-chatbot.git
   git push -u origin main
   ```
2. **Deploy on Streamlit**:
   - Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
   - Click **"New app"**.
   - Select your repository (`youtube-rag-chatbot`), branch (`main`), and set the main file path to `app.py`.
3. **Configure Secrets**:
   - In the Streamlit deployment settings, click **"Advanced settings"** -> **"Secrets"**.
   - Add your Gemini API key:
     ```toml
     GOOGLE_API_KEY = "your_actual_gemini_api_key"
     ```
   - Click **"Deploy!"**
4. Your application will be live at `https://<your-app-name>.streamlit.app`!
