import spacy
from langdetect import detect, LangDetectException
from typing import List
from config import settings

try:
    nlp = spacy.load("pt_core_news_lg")
except OSError:
    print("Modelo 'pt_core_news_lg' do spaCy não encontrado.")
    print("Execute: python -m spacy download pt_core_news_lg")
    nlp = None

def validate_language(text: str) -> None:
    try:
        detected_lang = detect(text)
        if detected_lang != settings.SUPPORTED_LANGUAGE:
            raise ValueError(
                f"Idioma não suportado. Detectado: '{detected_lang}'. "
                f"Apenas '{settings.SUPPORTED_LANGUAGE}' é aceito."
            )
    except LangDetectException:
        raise ValueError("Não foi possível determinar o idioma do texto.")


def process_text(text: str) -> List[str]:
    if nlp is None:
        raise RuntimeError("O modelo do spaCy não está carregado. Verifique a instalação.")

    doc = nlp(text.lower())

    lemmas = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and token.is_alpha
    ]
    
    return lemmas