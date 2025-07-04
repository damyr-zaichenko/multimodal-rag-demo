import requests
import xml.etree.ElementTree as ET
from typing import List
import os 


class SitemapLoader:
    NAMESPACE = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

    def __init__(self, root_sitemap_url: str, cache_dir: str = "cache"):
        self.root_sitemap_url = root_sitemap_url
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, url: str) -> str:
        filename = url.rstrip("/").split("/")[-1]
        return os.path.join(self.cache_dir, filename)

    def _load_or_download(self, url: str) -> str:
        cache_path = self._get_cache_path(url)
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read()
        response = requests.get(url)
        response.raise_for_status()
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        return response.text

    def fetch_sitemap_index(self) -> List[str]:
        content = self._load_or_download(self.root_sitemap_url)
        root = ET.fromstring(content)
        return [e.text for e in root.findall("ns:sitemap/ns:loc", self.NAMESPACE)]

    def fetch_urls_from_sitemap(self, sitemap_url: str) -> List[str]:
        content = self._load_or_download(sitemap_url)
        root = ET.fromstring(content)
        return [e.text for e in root.findall("ns:url/ns:loc", self.NAMESPACE)]

    def get_all_urls(self) -> List[str]:
        all_urls = []
        for sitemap_url in self.fetch_sitemap_index():
            all_urls.extend(self.fetch_urls_from_sitemap(sitemap_url))
        return all_urls


class URLFilter:
    def __init__(self, all_urls: List[str]):
        filtered_urls = [url for url in all_urls if url not in ('https://www.deeplearning.ai/the-batch/', 
                                                                'https://www.deeplearning.ai/the-batch/about/')]
        self.all_urls = filtered_urls

    def get_article_urls(self) -> List[str]:
        return [
            url for url in self.all_urls
            if '/the-batch/' in url
            and '/tag/' not in url
            and '/page/' not in url
            and '/issue-' not in url
        ]

    def get_issue_urls(self) -> List[str]:
        return [
            url for url in self.all_urls
            if '/the-batch/' in url
            and '/tag/' not in url
            and '/page/' not in url
            and '/issue-' in url
        ]