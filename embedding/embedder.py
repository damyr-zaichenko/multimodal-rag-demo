import numpy as np
from typing import Optional, List, Dict
from sentence_transformers import SentenceTransformer
from PIL import Image
from io import BytesIO
import requests


class ChunkEmbedder:
    """
    Embeds a single chunk with optional image.
    """
    def __init__(self,
                 text_model_name: str = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
                 image_model_name: str = 'sentence-transformers/clip-ViT-B-32',
                 use_image: bool = False):
        self.text_model = SentenceTransformer(text_model_name)
        self.use_image = use_image

        if use_image:
            self.image_model = SentenceTransformer(image_model_name)

    def embed_text(self, text: str) -> np.ndarray:
        return self.text_model.encode(text, normalize_embeddings=True)

    def embed_image(self, url: str) -> Optional[np.ndarray]:
        try:
            response = requests.get(url, timeout=5)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            return self.image_model.encode(img, normalize_embeddings=True)
        except Exception as e:
            print(f"[Image Error] Failed to load image from {url}: {e}")
            return None

    def embed_chunk(self, chunk: Dict) -> np.ndarray:
        text_vec = self.embed_text(chunk["text"])

        if self.use_image and chunk.get("type") == "text+image" and chunk.get("image_url"):
            image_vec = self.embed_image(chunk["image_url"])
            if image_vec is not None:
                return self.fuse_embeddings(text_vec, [image_vec])

        return text_vec

    def fuse_embeddings(self, text_vec: np.ndarray, image_vecs: List[np.ndarray]) -> np.ndarray:
        if not image_vecs:
            return text_vec
        image_avg = np.mean(image_vecs, axis=0)
        return 0.7 * text_vec + 0.3 * image_avg
    

# embedder = ChunkEmbedder()
# embedding = embedder.embed_chunk({
#     "article_url":"https://www.deeplearning.ai/the-batch/deepseek-r1-regains-open-weights-crown/",
#     "chunk_id":4,
#     "text":"“Scientific research brings the greatest benefit to the country where the work happens because (i) the new knowledge diffuses fastest within that country, and (ii) the process of doing research creates new talent for that nation.” Read Andrew’s full letterhere. Other top AI news and research stories we covered in depth: Anthropic releasednew Claude 4 Sonnet and Claude 4 Opus models, achieving top-tier performance in code generation benchmarks. Google unveiled a wave of AI updates at I/O, including the Veo 3 video generator, the compact Gemma 3n model, and enhancements to Gemini Pro and Ultra. Researchers behind DeepSeek detailed thetraining strategies and hardware infrastructureused to build their V3 and R1 models. A study found thatOpenAI’s GPT-4o can accurately identify verbatim excerptsfrom paywalled O’Reilly books, raising fresh questions about training data sources. Subscribe to Data Points"
#     ,"image_url":None,
#     "type":"text",
#     "title":"DeepSeek-R1 regains open-weights crown",
#     "description":"Researchers find critical vulnerability in GitHub MCP server",
#     "tags":[
#     "Data Points"
#     ]
# })
# print(embedding)