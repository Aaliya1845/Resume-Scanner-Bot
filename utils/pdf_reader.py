import PyPDF2


def extract_text_from_pdf(uploaded_file):
    """
    Extract text from an uploaded PDF file.

    Parameters
    ----------
    uploaded_file : UploadedFile
        Streamlit uploaded PDF file.

    Returns
    -------
    str
        Extracted text from all pages.
    """

    text = ""

    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)

        for page in pdf_reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    except Exception as e:
        return f"Error reading PDF: {str(e)}"

    return text


def get_pdf_statistics(uploaded_file):
    """
    Return basic PDF statistics.
    """

    stats = {
        "pages": 0,
        "title": "Unknown",
        "author": "Unknown"
    }

    try:
        uploaded_file.seek(0)

        pdf = PyPDF2.PdfReader(uploaded_file)

        stats["pages"] = len(pdf.pages)

        metadata = pdf.metadata

        if metadata:
            stats["title"] = metadata.title if metadata.title else "Unknown"
            stats["author"] = metadata.author if metadata.author else "Unknown"

        uploaded_file.seek(0)

    except:
        pass

    return stats


def validate_pdf(uploaded_file):
    """
    Validate uploaded PDF.
    """

    try:
        pdf = PyPDF2.PdfReader(uploaded_file)

        if len(pdf.pages) == 0:
            return False, "PDF has no pages."

        return True, "Valid PDF."

    except Exception:
        return False, "Invalid or corrupted PDF."
