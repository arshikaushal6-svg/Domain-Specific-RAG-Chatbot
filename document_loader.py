from io import BytesIO
from pypdf import PdfReader

def load_uploaded_pdfs(uploaded_files):
    records = []
    for uploaded_file in uploaded_files:
        reader = PdfReader(BytesIO(uploaded_file.getvalue()))
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                records.append({"text":text, "source":uploaded_file.name, "page":page_number})
    return records
