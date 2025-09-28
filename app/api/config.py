import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    
    HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    
    MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    HF_API_URL: str = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

    SUPPORTED_LANGUAGE: str = "pt"

settings = Settings()

if not settings.HUGGINGFACE_API_KEY:
    raise ValueError("A variável de ambiente HUGGINGFACE_API_KEY não foi encontrada. "
                     "Verifique se o arquivo .env existe e está configurado corretamente.")
