import spacy
from langdetect import detect, LangDetectException
from typing import List
from app.api.config import settings

try:
    nlp = spacy.load("pt_core_news_lg")
except OSError:
    print("Modelo 'pt_core_news_lg' do spaCy não encontrado.")
    print("Execute: python -m spacy download pt_core_news_lg")
    nlp = None

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