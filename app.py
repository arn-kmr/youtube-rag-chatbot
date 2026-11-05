import sys
import os
from pathlib import Path

# Add project root and src to sys.path for Streamlit Cloud compatibility
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st
from dotenv import load_dotenv, find_dotenv

try:
    from src.utils import extract_video_id, linkify_timestamps, fetch_video_metadata, parse_time_to_seconds
    from src.transcript_loader import fetch_transcript_documents, create_documents_from_text
    from src.vector_store import get_embedding_model, get_or_create_vector_store
    from src.rag_pipeline import build_rag_pipeline
except ImportError:
    from utils import extract_video_id, linkify_timestamps, fetch_video_metadata, parse_time_to_seconds
    from transcript_loader import fetch_transcript_documents, create_documents_from_text
    from vector_store import get_embedding_model, get_or_create_vector_store
    from rag_pipeline import build_rag_pipeline

# Load environment variables
load_dotenv(find_dotenv())

# Handle Streamlit Cloud secrets fallback
if not os.getenv("GOOGLE_API_KEY") and hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Configure Streamlit page
st.set_page_config(
    page_title="YouTube RAG AI Chatbot",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom modern CSS styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
        letter-spacing: -0.02em;
    }
    .sub-title {
        color: #6c757d;
        font-size: 1.0rem;
        margin-bottom: 1.2rem;
    }
    /* Sticky Left Column so Video Player stays pinned while scrolling */
    [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child {
        position: sticky !important;
        top: 1.5rem !important;
        align-self: flex-start !important;
        max-height: calc(100vh - 2.5rem);
        overflow-y: auto;
    }
    .video-info-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 15px;
    }
    .video-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #212529;
        margin-bottom: 4px;
        line-height: 1.3;
    }
    .video-meta {
        font-size: 0.85rem;
        color: #6c757d;
    }
    .badge-chip {
        display: inline-block;
        background-color: #e3f2fd;
        color: #0d47a1;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .source-box {
        background-color: #f8f9fa;
        border-left: 3px solid #1976d2;
        padding: 8px 12px;
        margin: 6px 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading embedding model into memory...")
def load_cached_embeddings():
    """Loads the sentence-transformers embedding model once across sessions."""
    return get_embedding_model()


# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None
if "current_video_url" not in st.session_state:
    st.session_state.current_video_url = None
if "video_meta" not in st.session_state:
    st.session_state.video_meta = {}
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "quick_prompt" not in st.session_state:
    st.session_state.quick_prompt = None
if "player_start_time" not in st.session_state:
    st.session_state.player_start_time = 0
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

embeddings = load_cached_embeddings()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("🎥 YouTube RAG")
    st.caption("Chat with any YouTube video with timestamped citations & hybrid search.")
    st.divider()

    st.subheader("1. Video Source")
    default_sample = "https://www.youtube.com/watch?v=Gfr50f6ZBvo"
    input_url = st.text_input(
        "Enter YouTube URL or Video ID:",
        value=st.session_state.current_video_url or default_sample,
        placeholder="https://www.youtube.com/watch?v=...",
    )

    col_proc, col_clear = st.columns([1.2, 1])
    with col_proc:
        process_clicked = st.button("🚀 Load Video", use_container_width=True, type="primary")
    with col_clear:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    with st.expander("📝 Custom / Pasted Transcript"):
        st.caption("Paste transcript text or notes directly if video has no captions:")
        manual_text = st.text_area("Paste Transcript Text:", height=120, placeholder="[00:15] Welcome everyone...\n[01:30] Today we discuss...")
        manual_id = st.text_input("Custom Video ID / Label:", value="custom_video")
        manual_submit = st.button("📥 Index Custom Transcript", use_container_width=True)

    st.divider()
    st.subheader("2. Model & Search Settings")
    selected_model = st.selectbox(
        "Gemini Model",
        options=["gemini-3.5-flash", "gemini-3.8-flash", "gemini-flash-latest"],
        index=0,
        help="Fast & high-capacity Gemini models for grounded QA.",
    )
    use_hybrid = st.toggle(
        "Hybrid Search (Dense + BM25)",
        value=True,
        help="Combines semantic vector search with keyword search (BM25) for superior accuracy on names, acronyms, and technical terms.",
    )
    top_k = st.slider("Context Chunks (k)", min_value=2, max_value=8, value=4)
    chunk_size = st.slider("Chunk Size (characters)", min_value=500, max_value=1500, value=900, step=100)

    st.divider()
    # API Key Configuration
    api_key_env = os.getenv("GOOGLE_API_KEY")
    if not api_key_env:
        user_key = st.text_input("Google AI API Key:", type="password", placeholder="AIzaSy...")
        if user_key:
            os.environ["GOOGLE_API_KEY"] = user_key
            st.success("API Key updated for this session!")

    # Chat Export Button in Sidebar
    if st.session_state.messages:
        chat_export_text = "# YouTube RAG Chat Transcript\n\n"
        if st.session_state.video_meta:
            chat_export_text += f"**Video:** {st.session_state.video_meta.get('title', '')}\n"
            chat_export_text += f"**URL:** {st.session_state.current_video_url}\n\n---\n\n"
        for m in st.session_state.messages:
            role_name = "User" if m["role"] == "user" else "Assistant"
            chat_export_text += f"### {role_name}:\n{m['content']}\n\n"

        st.download_button(
            label="💾 Download Chat (Markdown)",
            data=chat_export_text,
            file_name=f"chat_{st.session_state.current_video_id or 'notes'}.md",
            mime="text/markdown",
            use_container_width=True,
        )


# ----------------- VIDEO PROCESSING -----------------
# 1. Process custom manual transcript if submitted
if manual_submit and manual_text:
    try:
        with st.spinner("Processing custom transcript..."):
            docs = create_documents_from_text(manual_text, video_id=manual_id, chunk_size=chunk_size)
            vstore = get_or_create_vector_store(manual_id, docs, embeddings)
            st.session_state.messages = []
            st.session_state.current_video_id = manual_id
            st.session_state.current_video_url = f"https://www.youtube.com/watch?v={manual_id}"
            st.session_state.vector_store = vstore
            st.session_state.chunk_count = len(docs)
            st.session_state.video_meta = {
                "title": f"Custom Transcript ({manual_id})",
                "author": "User Ingested",
                "thumbnail_url": "",
            }
            st.toast("Custom transcript indexed successfully!", icon="✅")
            st.rerun()
    except Exception as e:
        st.error(f"Error indexing custom transcript: {e}")

# 2. Process YouTube URL
elif process_clicked and input_url:
    video_id = extract_video_id(input_url)
    if not video_id:
        st.error("Invalid YouTube URL or ID format. Please check and try again.")
    else:
        try:
            with st.spinner("Fetching transcript (API / yt-dlp) and building index..."):
                docs = fetch_transcript_documents(video_id, chunk_size=chunk_size)
                vstore = get_or_create_vector_store(video_id, docs, embeddings)
                metadata = fetch_video_metadata(video_id)

                # Reset chat if changing videos
                if st.session_state.current_video_id != video_id:
                    st.session_state.messages = []

                st.session_state.current_video_id = video_id
                st.session_state.current_video_url = f"https://www.youtube.com/watch?v={video_id}"
                st.session_state.vector_store = vstore
                st.session_state.video_meta = metadata
                st.session_state.chunk_count = len(docs)
                st.session_state.player_start_time = 0
                st.toast("Video transcript indexed successfully!", icon="✅")
                st.rerun()
        except Exception as e:
            st.error(f"Error loading video: {e}")


# ----------------- MAIN INTERFACE -----------------
st.markdown('<div class="main-title">YouTube Transcript AI Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Chat with any YouTube video with timestamped citations and interactive seek control.</div>',
    unsafe_allow_html=True,
)

if not st.session_state.vector_store or not st.session_state.current_video_id:
    st.info("👈 Enter a YouTube URL in the sidebar and click **🚀 Load Video** to begin.")
    st.stop()

# Layout: Split into Video Player (left) and Chat Area (right)
col_player, col_chat = st.columns([1, 1.25], gap="large")

with col_player:
    # Rich Video Info Card
    meta = st.session_state.video_meta or {}
    vtitle = meta.get("title", f"Video ({st.session_state.current_video_id})")
    vauthor = meta.get("author", "YouTube Creator")

    st.markdown(
        f"""
        <div class="video-info-card">
            <div class="video-title">🎬 {vtitle}</div>
            <div class="video-meta">
                <span class="badge-chip">👤 {vauthor}</span>
                <span class="badge-chip">📑 {st.session_state.chunk_count} Segments</span>
                <span class="badge-chip">🔍 {'Dense + BM25 Hybrid' if use_hybrid else 'FAISS Dense'}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Embedded Player with dynamic seek
    st.video(st.session_state.current_video_url, start_time=st.session_state.player_start_time)

    # Quick Prompts
    st.markdown("**⚡ Quick Prompts**")
    qp1, qp2 = st.columns(2)
    with qp1:
        if st.button("📋 Full Summary", use_container_width=True):
            st.session_state.quick_prompt = "Can you provide a comprehensive summary of the main points discussed in this video?"
        if qp2.button("💡 Key Takeaways", use_container_width=True):
            st.session_state.quick_prompt = "What are the most important takeaways, lessons, and insights from this video?"
    qp3, qp4 = st.columns(2)
    with qp3:
        if st.button("⏱️ Chapters Outline", use_container_width=True):
            st.session_state.quick_prompt = "Outline the major topics covered in this video along with approximate timestamps for each chapter."
    with qp4:
        if st.button("❓ Top Q&A", use_container_width=True):
            st.session_state.quick_prompt = "What are the top 5 questions answered in this video and their concise answers?"


with col_chat:
    st.subheader("💬 Chat with Video")

    # Display existing chat messages in a fixed-height scrollable container
    chat_container = st.container(height=580)
    with chat_container:
        for msg_idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg and msg["sources"]:
                    with st.expander("🔍 View Context Sources & Clickable Timestamps"):
                        for i, doc in enumerate(msg["sources"], 1):
                            ts = doc.get("timestamp", "N/A")
                            linkified_content = linkify_timestamps(doc.get("content", ""), st.session_state.current_video_id)
                            st.markdown(f"**Source {i} (Timestamp: {ts})**")
                            st.markdown(linkified_content)

                            # Quick jump in player button
                            if ts != "N/A":
                                secs = parse_time_to_seconds(ts)
                                if st.button(f"▶️ Seek Player to {ts}", key=f"seek_{msg_idx}_{i}"):
                                    st.session_state.player_start_time = secs
                                    st.rerun()

    # Input prompt handling docked right below the chat container
    user_input = st.chat_input("Ask a question about this video...")
    if st.session_state.quick_prompt:
        user_input = st.session_state.quick_prompt
        st.session_state.quick_prompt = None

    if user_input:
        # Display user and streaming assistant message directly inside the scrollable container
        st.session_state.messages.append({"role": "user", "content": user_input})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)

            # Assistant response
            with st.chat_message("assistant"):
                try:
                    pipeline = build_rag_pipeline(
                        vector_store=st.session_state.vector_store,
                        model_name=selected_model,
                        top_k=top_k,
                        use_hybrid=use_hybrid,
                    )

                    # Prior chat history (excluding current user turn)
                    history = st.session_state.messages[:-1]
                    stream_generator, retrieved_docs = pipeline(
                        question=user_input,
                        chat_history=history,
                        video_id=st.session_state.current_video_id,
                    )

                    # Stream response
                    raw_response = st.write_stream(stream_generator)

                    # Turn timestamps like [04:25] into clickable YouTube jump links
                    linkified_response = linkify_timestamps(raw_response, st.session_state.current_video_id)

                    # Format source documents
                    source_data = [
                        {
                            "timestamp": d.metadata.get("timestamp", "N/A"),
                            "content": d.page_content,
                        }
                        for d in retrieved_docs
                    ]

                    # Show sources expander
                    if source_data:
                        with st.expander("🔍 View Context Sources & Clickable Timestamps"):
                            for i, s in enumerate(source_data, 1):
                                st.markdown(f"**Source {i} (Timestamp: {s['timestamp']})**")
                                st.markdown(linkify_timestamps(s["content"], st.session_state.current_video_id))

                    # Save assistant response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": linkified_response,
                        "sources": source_data,
                    })

                    # Rerun if timestamps converted into markdown links so UI reflects formatting
                    if linkified_response != raw_response:
                        st.rerun()

                except Exception as err:
                    error_msg = f"An error occurred: {str(err)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
