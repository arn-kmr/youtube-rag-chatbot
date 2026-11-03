from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel,RunnableSequence,RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

######################################################################

#indexing1a(document ingestion)
video_id = "Gfr50f6ZBvo" # only the id not full url
try:
   

    api = YouTubeTranscriptApi()

    fetch_transcript = api.fetch(video_id)

    transcript = " ".join(
        snippet.text for snippet in fetch_transcript
    )

   # print(text)


except TranscriptsDisabled:
    print("no captions available for this video.")

######################################################################



######################################################################


# indexing1b(text splitting)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)    

chunks = splitter.create_documents([transcript])
#print(len(chunks))
######################################################################


######################################################################
#indexing 1c & 1d (embedding generation and storing in vector store)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

vector_store = FAISS.from_documents(chunks,embeddings)
#print(vector_store.index_to_docstore_id)

#####################################################################

# RETRIeVAL 

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":4})

#print(retriever.invoke("what is deepmind"))

######################################################################


#######################################################################
# AuGMENTATION

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer only from the provided transcript context.
      if the context is insufficient, just say you don't know.

      
      {context}
      question:{question}
    """,
    input_variables=['context','question']
)


question = "is the topic of aliens discussed in this video? if yes then what was discussed"

retrieved_docs = retriever.invoke(question)
#print(retrieved_docs)

#trimming all 4 documents into single string

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
   'context': retriever | RunnableLambda(format_docs),
   'question': RunnablePassthrough()
})


parser =  StrOutputParser()

main_chain = parallel_chain | prompt | model | parser

result = main_chain.invoke('can you summarize the video')


print(result)








######################################################################


