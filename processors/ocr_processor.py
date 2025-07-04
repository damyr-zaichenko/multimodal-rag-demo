import pytesseract
from PIL import Image
import requests
from io import BytesIO
import re


class OCRProcessor:
    """
    Handles OCR processing from image URLs and cleans extracted text.
    """

    def __init__(self, timeout: int = 5):
        """
        :param timeout: Timeout for image download requests.
        """
        self.timeout = timeout

    def extract_text_from_url(self, url: str) -> str:
        """
        Downloads an image from a URL and extracts raw OCR text.
        """
        try:
            response = requests.get(url, timeout=self.timeout)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            return pytesseract.image_to_string(img)
        except Exception as e:
            print(f"[OCR Error] Could not extract text from {url}: {e}")
            return ""

    def clean_text(self, text: str) -> str:
        """
        Cleans raw OCR text by removing artifacts, symbols, and normalizing whitespace.
        """
        # Remove non-printable characters and symbols
        text = re.sub(r"[^\x20-\x7E\n]", "", text)

        # Filter out lines mostly composed of symbols
        lines = text.splitlines()
        cleaned_lines = [
            line for line in lines
            if len(re.sub(r"[a-zA-Z0-9]", "", line)) < len(line) * 0.5
        ]

        # Normalize whitespace
        cleaned_text = " ".join(cleaned_lines)
        return re.sub(r"\s+", " ", cleaned_text).strip()

    def extract_and_clean(self, url: str) -> str:
        """
        Full OCR pipeline: extract raw text from image and clean it.
        """
        raw_text = self.extract_text_from_url(url)
        return self.clean_text(raw_text)