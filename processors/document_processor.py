from typing import Dict, List, Optional
from processors.ocr_processor import OCRProcessor  
from urllib.parse import urlparse
from nltk.tokenize import word_tokenize
import pytesseract
from PIL import Image
from io import BytesIO
import requests

class DocumentProcessor:
    def __init__(self, max_tokens: int = 200, tokenizer=None):
        """
        :param max_tokens: Maximum token length per chunk
        :param tokenizer: Optional tokenizer (e.g., from HuggingFace); if None, fallback to nltk.word_tokenize
        """
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.ocr = OCRProcessor()

    def normalize(self, raw: Dict[str, any]) -> Dict[str, any]:
        """
        Normalizes raw article or issue data into a unified document format.
        """
        return {
            "article_url": raw["url"],
            "title": raw.get("title", ""),
            "description": raw.get("description", ""),
            "tags": raw.get("tags", []),
            "blocks": raw["blocks"]
        }

    def chunk(self, doc: Dict[str, any]) -> List[Dict[str, any]]:
        """
        Splits a normalized document into text or text+image chunks.
        """
        chunks = []
        text_buf = ""
        chunk_id = 0
        last_image = None

        for block in doc["blocks"]:
            if block["type"] == "image":
                last_image = block["url"]
                ocr_text = self._run_ocr(last_image).strip()
                
                if ocr_text:
                    # Marker for image and its content
                    ocr_block = f"\n\n[ImageOCR:\n{ocr_text.strip()}]\n\n"
                    text_buf += " " + ocr_block
                else:
                    text_buf += f"\n[Image: {last_image}]\n"
                
                continue

            if block["type"] == "text":
                text_buf += " " + block["content"]
                token_len = self._count_tokens(text_buf)
                if token_len >= self.max_tokens:
                    chunks.append(self._make_chunk(doc, chunk_id, text_buf.strip(), last_image))
                    chunk_id += 1
                    text_buf = ""
                    last_image = None

        if text_buf.strip():
            chunks.append(self._make_chunk(doc, chunk_id, text_buf.strip(), last_image))

        return chunks

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(word_tokenize(text))

    def _make_chunk(
        self,
        doc: Dict[str, any],
        chunk_id: int,
        text: str,
        image_url: Optional[str]
    ) -> Dict[str, any]:
        return {
            "article_url": doc["url"],
            "chunk_id": chunk_id,
            "text": text,
            "image_url": image_url,
            "type": "text+image" if image_url else "text",
            "title": doc.get("title", ""),
            "description": doc.get("description", ""),
            "tags": doc.get("tags", [])
        }
    
    def _run_ocr(self, url: str) -> str:
        return self.ocr.extract_and_clean(url)