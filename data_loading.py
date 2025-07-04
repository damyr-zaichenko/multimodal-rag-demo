import os
import time
import streamlit as st
from scrapers.sitemap_loader import SitemapLoader, URLFilter
from scrapers.article_scrapper import ArticleScraper
from scrapers.issue_article_scrapper import IssueArticleScraper
from embedding.embedder import ChunkEmbedder
from storage.parsed_article_store import ParsedArticleStore
from storage.faiss_chunk_store import FaissChunkStore
from processors.document_processor import DocumentProcessor
from utils.logger import setup_logger


class BatchDataPipeline:
    """
    Handles the ingestion and embedding of articles
    from The Batch newsletter.
    
    Stages:
        1. Scraping articles from sitemap and saving as JSON.
        2. Embedding article chunks and storing in FAISS.
    """

    def __init__(self,
                 sitemap_url: str = "https://www.deeplearning.ai/sitemap.xml",
                 parsed_store_path: str = 'data/parsed_articles.json',
                 faiss_dim: int = 384):
        self.logger = setup_logger(self.__class__.__name__)
        self.sitemap_url = sitemap_url
        self.store = ParsedArticleStore(parsed_store_path)
        self.faiss_store = FaissChunkStore(dim=faiss_dim)
        self.embedder = ChunkEmbedder()
        self.document_processor = DocumentProcessor(max_tokens=200)

    def parse_and_store_articles(self, limit: int = 4000):
        """
        Downloads articles from sitemap, parses content, and stores in local JSON.
        Avoids duplicates already saved.
        """
        loader = SitemapLoader(self.sitemap_url)
        all_urls = loader.get_all_urls()

        filterer = URLFilter(all_urls)
        article_urls = filterer.get_article_urls()[:limit]
        issue_urls = filterer.get_issue_urls()[:limit]

        all_targets = issue_urls + article_urls
        already_loaded = {a['url'] for a in self.store.load()}

        for url in all_targets:
            if url in already_loaded:
                continue

            scraper = ArticleScraper(url)
            parsed_article = scraper.parse()

            if parsed_article:
                self.store.add_article(parsed_article)
                self.store.save()
                st.text(f"Parsed: {url}")
                self.logger.info(f"Parsed article: {url}")

            time.sleep(0.3)  

    def embed_and_index_articles(self):
        """
        Embeds parsed articles and stores vectors in FAISS.
        Skips already embedded URLs.
        """
        articles = self.store.load()
        embedded_urls = set(self.faiss_store.get_all_urls())

        self.logger.info(f"Embedding {len(articles)} articles...")

        for article in articles:
            if article['url'] in embedded_urls:
                continue

            chunks = self.document_processor.chunk(article)
            for chunk in chunks:
                st.json(chunk)
                embedding = self.embedder.embed_chunk(chunk)
                self.faiss_store.add(embedding, metadata=chunk)

            self.faiss_store.save()
            self.logger.info(f"Added {len(chunks)} chunks from {article['url']}")


if __name__ == "__main__":
    pipeline = BatchDataPipeline()

    # Run both stages
    pipeline.parse_and_store_articles()
    pipeline.embed_and_index_articles()