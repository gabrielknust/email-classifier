from fastapi import APIRouter, HTTPException,UploadFile, File
import httpx
from ..models.api_models import EmailRequest, ClassificationResponse
from .pipeline import run_classification_pipeline
from app.api.src.controller import file_reader

router = APIRouter()

@router.post(
    "/process-email",
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
    
@router.post("/process-file", response_model=ClassificationResponse, tags=["Processamento de E-mail"])
async def process_file_endpoint(file: UploadFile = File(...)):
    try:
        # 1. Usa o file_reader para extrair o texto do arquivo
        text_content = file_reader.read_text_from_file(file)
        if not text_content.strip():
            raise ValueError("O arquivo parece estar vazio ou não contém texto extraível.")
        
        # 2. Envia o texto extraído para a mesma pipeline
        final_result = await run_classification_pipeline(text=text_content)
        return final_result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro interno: {str(e)}")