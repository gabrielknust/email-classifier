from fastapi import APIRouter, HTTPException
import httpx
from ..models.api_models import EmailRequest, ClassificationResponse
from .pipeline import run_classification_pipeline

router = APIRouter()

@router.post(
    "/classify",
    response_model=ClassificationResponse,
    tags=["Classification"],
    summary="Classifica um texto de e-mail em Produtivo ou Improdutivo"
)
async def classify_text_endpoint(request: EmailRequest):
    try:
        result_dict = await run_classification_pipeline(text=request.text)
        return ClassificationResponse(**result_dict)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Ocorreu um erro interno no servidor.")