import requests
from bs4 import BeautifulSoup
from typing import List, Dict
from urllib.parse import urlparse, parse_qs, unquote
import time

class ArticleScraper:
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0; +https://example.com/bot)"
    }
    def __init__(self, url: str, max_retries: int = 3, retry_delay: float = 2.0):
        self.url = url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.soup = self._load_soup()

    def _load_soup(self) -> BeautifulSoup:
        for attempt in range(self.max_retries):
            try:
                response = requests.get(self.url, headers=self.HEADERS, timeout=10)
                response.raise_for_status()
                return BeautifulSoup(response.content, "html.parser")
            except requests.HTTPError as e:
                print(f"[{attempt+1}/{self.max_retries}] HTTPError: {e} for URL: {self.url}")
            except requests.RequestException as e:
                print(f"[{attempt+1}/{self.max_retries}] Request failed: {e} for URL: {self.url}")
            time.sleep(self.retry_delay)

        raise RuntimeError(f"Failed to load page after {self.max_retries} attempts: {self.url}")

    def get_title_and_description(self) -> Dict[str, str]:
        h1 = self.soup.find("h1", class_="mt-6 leading-tight text--l1 text-slate-800")
        if not h1:
            return {"title": "", "description": ""}

        title = h1.contents[0].strip() if h1.contents else ""
        description = h1.find("span")
        desc_text = description.get_text(strip=True) if description else ""
        return {"title": title, "description": desc_text}

    def get_tags(self) -> List[str]:
        tag_container = self.soup.find("div", class_="flex flex-wrap gap-2 mt-8")
        if not tag_container:
            return []
        return [tag.get_text(strip=True) for tag in tag_container.find_all("div")]

    # def get_article_text(self) -> str:
    #     content = self.soup.find("div", class_="prose--styled justify-self-center post_postContent__wGZtc")
    #     if not content:
    #         return ""
    #     return "\n".join([p.get_text(strip=True) for p in content.find_all(["p", "li"])])

    # def parse(self) -> Dict[str, any]:
    #     return {
    #         "url": self.url,
    #         **self.get_title_and_description(),
    #         "tags": self.get_tags(),
    #         "text": self.get_article_text(),
    #         "images": [self.get_main_image()]
    #     }
        
    def get_main_image(self) -> str:
        for img in self.soup.find_all("img"):
            src = img.get("src", "")
            if not src.startswith("/_next/image/?url="):
                continue

            parsed = urlparse(src)
            qs = parse_qs(parsed.query)
            encoded_url = qs.get("url", [""])[0]
            decoded_url = unquote(encoded_url)

            # === Filter rules ===
            if "logo" in decoded_url.lower():
                continue
            if decoded_url.startswith("/_next/static"):
                continue
            if "home-wordpress.deeplearning.ai" in decoded_url:
                continue
            if img.get("data-sentry-source-file") == "Advertisement.tsx":
                continue
            if img.get("alt", "").strip() == "":
                continue

            return decoded_url

        return ""
    
    def get_article_blocks(self) -> List[Dict[str, str]]:
        """
        Returns a linear sequence of blocks from the article:
        - {"type": "text", "content": "..."}
        - {"type": "image", "url": "..."}
        """
        content = self.soup.find("div", class_="prose--styled justify-self-center post_postContent__wGZtc")
        if not content:
            return []

        blocks = []
        added_main_image = False
        main_image = self.get_main_image()

        for el in content.find_all(["p", "li"], recursive=True):
            text = el.get_text(strip=True)
            if text:
                blocks.append({"type": "text", "content": text})
                # Insert image after first non-empty text block
                if not added_main_image and main_image:
                    blocks.append({"type": "image", "url": main_image})
                    added_main_image = True

        return blocks
    
    def parse(self) -> Dict[str, any]:
        return {
            "url": self.url,
            **self.get_title_and_description(),
            "tags": self.get_tags(),
            "blocks": self.get_article_blocks()
        }
    