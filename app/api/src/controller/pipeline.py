from typing import Dict, List, Any
from . import preprocessor
from . import classifier

def run_classification_pipeline(text: str, labels: List[str]) -> Dict[str, Any]:
    preprocessor.validate_language(text)
    
    processed_tokens = preprocessor.process_text(text)
    print(f" -> Tokens processados: {processed_tokens}")

    processed_text_string = " ".join(processed_tokens)
    print(f" -> Texto transformado para envio: '{processed_text_string}'")

    classification_result = classifier.classify_text(text=processed_text_string, labels=labels)

    return classification_result