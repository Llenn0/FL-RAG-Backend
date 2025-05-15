import io
import json
import os
import ollama
import pdfplumber
from langchain_core.load import dumps, dumpd
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_ollama.llms import OllamaLLM
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

from prompt import CONTEXT_PROMPT, PROMPT_TEMPLATE

chat_history = []

# Function for performing contextualised chunking strategy. Stores the contextual chunks in S3 database for later use.
def chunk_and_store(file_bytes, filename, s3_client):
    # Create a list of Documents from the provided files - bytes requires some conversion into correct format
    docs = []
    with pdfplumber.open(file_bytes) as pdf_file:
        for page in pdf_file.pages:
            docs.append(Document(page.extract_text(), metadata={'page': len(docs)}))

    # Split each document to create chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    split_doc = splitter.split_documents(docs)
    print("Successfully split docs")

    llm = OllamaLLM(model="llama3.2")
    chunk_no = 0
    # Contextualise the chunks using LLM
    for chunk in split_doc:
        # Format chunk as a prompt for document analysis LLM
        message = CONTEXT_PROMPT.format_messages(doc=docs[chunk.metadata['page']].page_content,
                                                 chunk=chunk.page_content)
        response = llm.invoke(message)

        # Add the returned context to the chunk as initial page content,
        chunk.page_content = f"{response}\n\n{chunk.page_content}"
        # Convert back to bytes for upload to S3
        chunk_dict = dumpd(chunk)
        chunk_bytes = io.BytesIO(json.dumps(chunk_dict).encode('utf-8'))

        # Upload S3 using boto3 API
        s3_client.upload_fileobj(
            chunk_bytes,
            os.getenv("AWS_BUCKET_NAME"),
            f"chunks/{filename}-{str(chunk_no)}.json"
        )
        print(f"Chunk upload success for {filename}-{str(chunk_no)}")
        chunk_no += 1


# Given a question and list of chunks, performs retrieval task and returns LLM response
def generate_response(input_question, chunks_contextual):
    embeddings = OllamaEmbeddings(model="llama3.2") # Vector embeddings to use
    # We use a semantic + bm25 ensemble approach to improve retrieval accuracy
    # Create vector embeddings and define a retriever
    semantic_retriever = InMemoryVectorStore.from_documents(chunks_contextual, embeddings).as_retriever(kwargs={"k": 4})
    bm25_retriver = BM25Retriever.from_documents(chunks_contextual, k=4)
    ensemble = EnsembleRetriever(retrievers=[semantic_retriever, bm25_retriver], weights=[0.5, 0.5])

    # Define a reranker for final ranking
    reranker = FlashrankRerank(model='ms-marco-MiniLM-L-12-v2', top_n=5)

    # Final retriever object defines the entire process
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=ensemble
    )

    # Invoke the retriever and append the relevance score to each relevant chunk
    context = compression_retriever.invoke(input_question)
    for i in range(0, len(context)):
        rel = context[i].metadata["relevance_score"]
        rel_doc = ''.join(("[Relevance Score: ", str(rel), "] ", context[i].page_content))
        context[i].page_content = rel_doc

    # Define the LLM for final response
    ollama.pull(model="deepseek-r1:7b")
    model = OllamaLLM(model="deepseek-r1:7b")

    # Format the question according to the prompt template, inserting context and chat history
    message = PROMPT_TEMPLATE.format_messages(context=context, question=input_question, chat_history=chat_history)

    # Invoke the model and remove the thinking element so that the user does not see it
    response = model.invoke(message)
    answer = response.split('</think>')[1]

    # Update chat history
    chat_history.append(HumanMessage(content=input_question))
    chat_history.append(AIMessage(content=answer))

    return answer
