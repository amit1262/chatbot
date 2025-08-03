"""
Initialize the model using environment variables.
"""

import os
from langchain.chat_models import init_chat_model

# model initialization
language_model = None
MODEL_NAME = os.environ.get("LANGUAGE_MODEL_NAME")
try:
    language_model = init_chat_model(
        model=MODEL_NAME, model_provider=os.environ.get("LANGUAGE_MODEL_PROVIDER")
    )
    assert language_model, f"Failed to initialize {MODEL_NAME} model"
    print(f"{MODEL_NAME} initialized successfully.")
except Exception as e:
    raise RuntimeError(f"Error initializing model: {MODEL_NAME}: {e}")
