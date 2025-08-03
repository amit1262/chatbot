import os
from langchain_openai.embeddings import AzureOpenAIEmbeddings

# embedding model
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", None)
assert (
    EMBEDDING_MODEL_NAME is not None
), "EMBEDDING_MODEL_NAME environment variable is not set."

try:
    embedding_model = AzureOpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
except Exception as e:
    raise RuntimeError(f"Error initializing model: {EMBEDDING_MODEL_NAME}: {e}")
