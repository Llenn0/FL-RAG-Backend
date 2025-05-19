# FL_RAG_Backend

This repository contains a Python Flask API implementing a RAG chatbot solution. The API's features can be split into two categories:
- Interaction with AWS S3 via the Boto3 API. This allows users to upload PDF documents to storage, where they are automatically converted and chunked for use in the RAG system. It also provides functionality to list the names of all files currently stored.
- AI Chatbot response generation using Langchain. Stored documents are used as context, and the API performs a retrieval step to gather the most relevant information to guide the LLM's response.

The API is a minimal Proof of Concept, which could be greatly improved by the addtion of other features such as deleting documents from storage, or improved RAG via GraphRAG. 
