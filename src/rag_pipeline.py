import os
from typing import List, Generator, Optional
from dotenv import load_dotenv, find_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

from src.vector_store import build_retriever
from src.utils import linkify_timestamps

load_dotenv(find_dotenv())

SYSTEM_PROMPT = """You are an intelligent YouTube Video Assistant.
Your task is to answer user questions accurately based ONLY on the provided transcript context from the video.

Guidelines:
1. Ground your answer strictly in the provided transcript context.
2. If the topic was not discussed or the context is insufficient, explicitly state: "I couldn't find information about that in this video." Do not hallucinate or extrapolate.
3. Whenever citing a specific topic, fact, quote, or discussion point, include the timestamp citation in brackets like `[MM:SS]` (or `[HH:MM:SS]`) from the provided context so the user can jump to that exact moment.
4. Keep answers concise, highly informative, and formatted with clean Markdown bullet points where appropriate.

Transcript Context:
{context}
"""


def format_retrieved_docs(docs: List[Document]) -> str:
    """Formats retrieved Document chunks into a structured context block."""
    formatted = []
    for doc in docs:
        ts = doc.metadata.get("timestamp", "N/A")
        content = doc.page_content.strip()
        formatted.append(f"--- Segment (around {ts}) ---\n{content}")
    return "\n\n".join(formatted)


def convert_chat_history(raw_history: List[dict]) -> List[BaseMessage]:
    """Converts raw session history list into LangChain BaseMessage objects."""
    messages = []
    for turn in raw_history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        elif turn["role"] == "assistant":
            messages.append(AIMessage(content=turn["content"]))
    return messages


def build_rag_pipeline(
    vector_store: FAISS,
    model_name: str = "gemini-3.5-flash",
    top_k: int = 4,
    use_hybrid: bool = True,
    api_key: Optional[str] = None,
):
    """
    Constructs a RAG pipeline with Hybrid Search (FAISS + BM25),
    chat history awareness, and streaming output.
    """
    retriever = build_retriever(
        vector_store=vector_store,
        top_k=top_k,
        use_hybrid=use_hybrid,
    )

    resolved_api_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not resolved_api_key:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
                resolved_api_key = st.secrets["GOOGLE_API_KEY"]
        except Exception:
            pass

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.2,
        google_api_key=resolved_api_key,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    parser = StrOutputParser()
    chain = prompt | llm | parser

    def query_stream(
        question: str,
        chat_history: List[dict],
        video_id: str = "",
    ) -> tuple[Generator[str, None, None], List[Document]]:
        """
        Retrieves relevant context chunks and returns an output stream
        along with the retrieved documents.
        """
        retrieved_docs = retriever.invoke(question)
        context_str = format_retrieved_docs(retrieved_docs)
        history_messages = convert_chat_history(chat_history)

        stream = chain.stream(
            {
                "context": context_str,
                "chat_history": history_messages,
                "question": question,
            }
        )

        return stream, retrieved_docs

    return query_stream
