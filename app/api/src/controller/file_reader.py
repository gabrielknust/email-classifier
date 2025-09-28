import io
from fastapi import UploadFile
from PyPDF2 import PdfReader

def read_text_from_file(file: UploadFile) -> str:
    """Lê o conteúdo de texto de um arquivo .txt ou .pdf."""
    filename = file.filename
    if filename.endswith(".txt"):
        content_bytes = file.file.read()
        return content_bytes.decode("utf-8")
    elif filename.endswith(".pdf"):
        pdf_stream = io.BytesIO(file.file.read())
        reader = PdfReader(pdf_stream)
        text_parts = [page.extract_text() for page in reader.pages]
        return "\n".join(text_parts)
    else:
        raise ValueError("Formato de arquivo não suportado. Por favor, envie .txt ou .pdf.")