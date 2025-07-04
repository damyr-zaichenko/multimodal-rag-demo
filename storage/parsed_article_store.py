import json
from typing import Dict, Any, List
from pathlib import Path
from utils.logger import setup_logger

logger = setup_logger(__name__)

class ParsedArticleStore:
    def __init__(self, path: str = "data/parsed_articles.json"):
        self.path = Path(path)
        self.articles: List[Dict[str, Any]] = []

        if self.path.exists():
            with open(self.path, "r") as f:
                self.articles = json.load(f)
            logger.info(f"Loaded {len(self.articles)} articles from {self.path}")
        else:
            logger.info(f"No existing file at {self.path}, starting fresh.")

    def find_index_by_url(self, url: str) -> int:
        """Return index of article with matching URL, or -1 if not found."""
        for i, item in enumerate(self.articles):
            if item.get("url") == url:
                return i
        return -1

    def add_article(self, article_data: Dict[str, Any], update: bool = False) -> bool:
        """
        Add or update an article (expects 'blocks', 'title', 'description', 'tags', and 'url').
        Returns True if added or updated, False if duplicate and update=False.
        """
        required_keys = {"url", "blocks"}
        if not required_keys.issubset(article_data.keys()):
            logger.warning(f"Invalid article data, missing keys: {required_keys - set(article_data)}")
            return False

        idx = self.find_index_by_url(article_data["url"])
        if idx >= 0:
            if update:
                self.articles[idx] = article_data
                logger.info(f"Updated article with URL: {article_data['url']}")
                return True
            logger.info(f"Skipped duplicate article with URL: {article_data['url']}")
            return False

        self.articles.append(article_data)
        logger.info(f"Added new article with URL: {article_data['url']}")
        return True

    def save(self) -> None:
        """Save all stored articles to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.articles, f, indent=2)
        logger.info(f"Saved {len(self.articles)} articles to {self.path}")

    def load(self) -> List[Dict[str, Any]]:
        """Load parsed articles from memory."""
        return self.articles