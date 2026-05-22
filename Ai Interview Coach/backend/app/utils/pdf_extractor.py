import io
import logging

log = logging.getLogger("uvicorn.error")


def extract_text_from_pdf(content: bytes) -> str:
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(content)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        result = "\n".join(pages).strip()
        if result:
            return result
    except Exception as e:
        log.warning("pdfplumber extraction failed: %s", e)

    try:
        import PyPDF2

        reader = PyPDF2.PdfReader(io.BytesIO(content))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        result = "\n".join(pages).strip()
        if result:
            return result
    except Exception as e:
        log.warning("PyPDF2 extraction failed: %s", e)

    try:
        from pdfminer.high_level import extract_text as pdfminer_extract

        result = pdfminer_extract(io.BytesIO(content)).strip()
        if result:
            return result
    except Exception as e:
        log.warning("pdfminer extraction failed: %s", e)

    log.error("All PDF extraction methods failed")
    return ""
