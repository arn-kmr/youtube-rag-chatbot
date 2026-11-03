import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".faiss_cache")


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Initializes and returns the HuggingFace embedding model."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def get_or_create_vector_store(
    video_id: str,
    documents: Optional[List[Document]],
    embeddings: HuggingFaceEmbeddings,
) -> FAISS:
    """
    Returns a FAISS vector store. If an index already exists on disk
    for this video_id, it is loaded directly to save computation.
    Otherwise, a new vector store is created and cached to disk.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    video_cache_path = os.path.join(CACHE_DIR, video_id)

    if os.path.exists(os.path.join(video_cache_path, "index.faiss")):
        return FAISS.load_local(
            video_cache_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    if not documents:
        raise ValueError("Cannot build vector store without documents.")

    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(video_cache_path)
    return vector_store


def build_retriever(
    vector_store: FAISS,
    top_k: int = 4,
    use_hybrid: bool = True,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
):
    """
    Constructs either a Hybrid Retriever (Dense FAISS + Sparse BM25 via EnsembleRetriever)
    or a standard Dense FAISS retriever.
    """
    dense_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    if not use_hybrid:
        return dense_retriever

    # Extract indexed documents from FAISS docstore for BM25
    all_docs = list(vector_store.docstore._dict.values())
    if not all_docs:
        return dense_retriever

    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = top_k

    ensemble = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[dense_weight, sparse_weight],
    )
    return ensemble
