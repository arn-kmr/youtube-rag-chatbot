
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1. Initialize Gemini
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash"
)

# 2. Document ingestion: fetch YouTube transcript
video_id = "Gfr50f6ZBvo"

try:
    api = YouTubeTranscriptApi()
    fetched_transcript = api.fetch(video_id)

    transcript = " ".join(
        snippet.text for snippet in fetched_transcript
    )

    if not transcript.strip():
        raise ValueError("The transcript is empty.")

except TranscriptsDisabled:
    raise SystemExit("Captions are disabled for this video.")

# 3. Text splitting
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.create_documents([transcript])

print("Number of chunks:", len(chunks))

# 4. Generate embeddings and create FAISS vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

# 5. Retrieval
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# 6. Augmentation: create prompt
prompt = PromptTemplate(
    template="""
You are a helpful assistant.
Answer only from the provided transcript context.
If the context is insufficient, say you don't know.

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

# 7. Format retrieved documents into one context string
def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )

# 8. Create parallel chain
parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

# 9. Build the complete RAG chain
parser = StrOutputParser()

main_chain = parallel_chain | prompt | model | parser

# 10. Ask a question
question = "Can you summarize the video?"

result = main_chain.invoke(question)

print("\nAnswer:")
print(result)
