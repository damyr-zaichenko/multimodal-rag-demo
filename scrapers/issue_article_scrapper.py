import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, unquote, urljoin
from urllib.parse import urljoin
from typing import Dict, List
import time

class IssueArticleScraper:
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0; +https://example.com/bot)"
    }

    def __init__(self, url: str, max_retries: int = 3, retry_delay: float = 2.0):
        self.url = url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.soup = self._load_soup()

    def _load_soup(self) -> BeautifulSoup:
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(self.url, headers=self.HEADERS, timeout=10)
                response.raise_for_status()
                return BeautifulSoup(response.content, "html.parser")
            except requests.HTTPError as e:
                print(f"[{attempt}/{self.max_retries}] HTTPError for {self.url}: {e}")
            except requests.RequestException as e:
                print(f"[{attempt}/{self.max_retries}] RequestException for {self.url}: {e}")
            time.sleep(self.retry_delay)

        raise RuntimeError(f"Failed to load issue page after {self.max_retries} attempts: {self.url}")

    def get_article_blocks(self) -> List[Dict[str, str]]:
        content = self.soup.find("div", class_="prose--styled justify-self-center post_postContent__wGZtc")
        if not content:
            return []

        blocks = []
        for el in content.find_all(["p", "li", "img"], recursive=True):
            if el.name in ["p", "li"]:
                text = el.get_text(strip=True)
                if text:
                    blocks.append({"type": "text", "content": text})
            elif el.name == "img":
                src = el.get("src")
                if not src:
                    continue
                # ✅ Handle Next.js proxy image URLs
                if src.startswith("/_next/image"):
                    parsed = urlparse(src)
                    query = parse_qs(parsed.query)
                    encoded_url = query.get("url", [""])[0]
                    decoded_url = unquote(encoded_url)
                    if decoded_url:
                        blocks.append({"type": "image", "url": decoded_url})
                else:
                    blocks.append({"type": "image", "url": urljoin(self.url, src)})

        return blocks

    def parse(self) -> Dict[str, any]:
        return {
            "url": self.url,
            "blocks": self.get_article_blocks()
        }