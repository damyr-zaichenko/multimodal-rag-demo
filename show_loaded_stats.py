from storage.faiss_chunk_store import FaissChunkStore
from storage.parsed_article_store import ParsedArticleStore

parsed_storage = ParsedArticleStore('data/parsed_articles.json')

parsed_urls = list(set([a['url'] for a in parsed_storage.load()]))

print(f'parsed articles: {len(parsed_urls)}')

encoded_storage = FaissChunkStore(dim=384)

encoded_urls = list(set(encoded_storage.get_all_urls()))

print(f'encoded articles {len(encoded_urls)}')

# print(len([url for url in parsed_urls if url not in encoded_urls]))
# print(len([url for url in parsed_urls if url in encoded_urls]))