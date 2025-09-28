import httpx
from typing import Dict, List, Any
from app.api.config import settings

def classify_text(text: str, labels: List[str]) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}"
    }

    payload = {
        "inputs": text,
        "parameters": {
            "candidate_labels": labels
        }
    }

    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.post(
                url=settings.HF_API_URL,
                headers=headers,
                json=payload
            )

        response.raise_for_status()
        return response.json()

    except httpx.HTTPStatusError as e:
        print(f"Erro da API: {e.response.status_code} - {e.response.text}")
        raise

    except httpx.RequestError as e:
        print(f"Erro de conexão: {e}")
        raise