from fastapi import APIRouter, HTTPException
import httpx
from app.api.src.models.api_models import ClassificationRequest, ClassificationResponse
from app.api.src.controller.pipeline import run_classification_pipeline

router = APIRouter()

@router.post(
    "/classify-text", 
    response_model=ClassificationResponse,
    tags=["Classification"]
)

async def classify_text_endpoint(request: ClassificationRequest):
    try:
        text_to_classify = request.text
        labels = request.labels
        
        result_dict = run_classification_pipeline(text=text_to_classify, labels=labels)

        return ClassificationResponse(**result_dict)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Erro na comunicação com o serviço de IA: {e.response.text}")
    except httpx.RequestError:
        raise HTTPException(status_code=504, detail="Erro de conexão com o serviço de IA.")
    except Exception as e:
        print(f"ERRO INTERNO INESPERADO: {e}")
        raise HTTPException(status_code=500, detail="Ocorreu um erro interno no servidor.")