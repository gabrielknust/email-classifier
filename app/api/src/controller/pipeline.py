from typing import Dict, Any
from app.api.src.controller.classifier import predict
from app.api.src.controller.preprocessor import process_text
from app.api.src.controller.response_generator import suggest_reply

async def run_classification_pipeline(text: str) -> Dict[str, Any]:

    text_preprocessed = ' '.join(process_text(text))
    
    classification_result = predict(text_preprocessed)

    suggested_reply = await suggest_reply(
        original_text=text, 
        label=classification_result['label']
    )

    final_result = {
        "label": classification_result["label"],
        "confidence": classification_result["confidence"],
        "suggested_reply": suggested_reply
    }

    return final_result