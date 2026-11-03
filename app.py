import streamlit as st
from dotenv import load_dotenv, find_dotenv

from src.utils import extract_video_id, linkify_timestamps
from src.transcript_loader import fetch_transcript_documents
from src.vector_store import get_embedding_model, get_or_create_vector_store
from src.rag_pipeline import build_rag_pipeline

# Load environment variables
load_dotenv(find_dotenv())

# Configure Streamlit page
st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .badge-video {
        background-color: #f0f2f6;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-family: monospace;
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
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "quick_prompt" not in st.session_state:
    st.session_state.quick_prompt = None

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

    st.divider()
    st.subheader("2. Model & Search Settings")
    selected_model = st.selectbox(
        "Gemini Model",
        options=["gemini-3.5-flash", "gemini-3.8-flash", "gemini-flash-latest"],
        index=0,
    )
    use_hybrid = st.toggle(
        "Hybrid Search (Dense + BM25)",
        value=True,
        help="Combines semantic vector search with keyword search (BM25) for superior accuracy on names and technical terms.",
    )
    top_k = st.slider("Context Chunks (k)", min_value=2, max_value=8, value=4)
    chunk_size = st.slider("Chunk Size (characters)", min_value=500, max_value=1500, value=900, step=100)

    st.divider()
    if st.session_state.current_video_id:
        st.success(f"Active Video: `{st.session_state.current_video_id}`")
        badge_text = "Dense + BM25 Hybrid" if use_hybrid else "Dense FAISS"
        st.caption(f"Retriever: **{badge_text}** (FAISS cached)")
    else:
        st.info("Enter a YouTube URL and click **Load Video** to begin.")


# ----------------- VIDEO PROCESSING -----------------
if process_clicked and input_url:
    video_id = extract_video_id(input_url)
    if not video_id:
        st.error("Invalid YouTube URL or ID format. Please check and try again.")
    else:
        try:
            with st.spinner("Fetching transcript (or audio fallback) and building index..."):
                docs = fetch_transcript_documents(video_id, chunk_size=chunk_size)
                vstore = get_or_create_vector_store(video_id, docs, embeddings)

                # Reset chat if changing videos
                if st.session_state.current_video_id != video_id:
                    st.session_state.messages = []

                st.session_state.current_video_id = video_id
                st.session_state.current_video_url = f"https://www.youtube.com/watch?v={video_id}"
                st.session_state.vector_store = vstore
                st.toast("Video transcript indexed successfully!", icon="✅")
                st.rerun()
        except Exception as e:
            st.error(f"Error loading video: {e}")


# ----------------- MAIN INTERFACE -----------------
st.markdown('<div class="main-title">YouTube Transcript AI Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Ask questions about any video. Answers are grounded in the transcript with interactive timestamp jump links.</div>',
    unsafe_allow_html=True,
)

if not st.session_state.vector_store or not st.session_state.current_video_id:
    st.info("👈 Paste a YouTube video URL in the sidebar and click **Load Video** to start chatting.")
    st.stop()

# Layout: Split into Video Player (left) and Chat Area (right)
col_player, col_chat = st.columns([1, 1.3], gap="large")

with col_player:
    st.subheader("📺 Video Player")
    st.video(st.session_state.current_video_url)

    st.markdown("**⚡ Quick Prompts**")
    q1, q2, q3 = st.columns(3)
    if q1.button("📋 Summarize", use_container_width=True):
        st.session_state.quick_prompt = "Can you provide a comprehensive summary of the main points discussed in this video?"
    if q2.button("💡 Key Takeaways", use_container_width=True):
        st.session_state.quick_prompt = "What are the key takeaways and highlights from this video?"
    if q3.button("❓ Topics Outline", use_container_width=True):
        st.session_state.quick_prompt = "What are the major topics covered in this video along with their approximate timestamps?"


with col_chat:
    st.subheader("💬 Chat with Video")

    # Display existing chat messages
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg and msg["sources"]:
                    with st.expander("🔍 View Context Sources & Clickable Timestamps"):
                        for i, doc in enumerate(msg["sources"], 1):
                            ts = doc.get("timestamp", "N/A")
                            linkified_content = linkify_timestamps(doc.get("content", ""), st.session_state.current_video_id)
                            st.markdown(f"**Source {i} (Timestamp: {ts})**")
                            st.markdown(linkified_content)

    # Input prompt handling (either from chat_input or quick buttons)
    user_input = st.chat_input("Ask a question about this video...")
    if st.session_state.quick_prompt:
        user_input = st.session_state.quick_prompt
        st.session_state.quick_prompt = None

    if user_input:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
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

                # Prior chat history (excluding the current user question)
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

                # Rerun if timestamps were converted into links so UI reflects clickable formatting
                if linkified_response != raw_response:
                    st.rerun()

            except Exception as err:
                error_msg = f"An error occurred: {str(err)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
