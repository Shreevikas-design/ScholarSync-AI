from __future__ import annotations

from config import Settings


class EmbeddingService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.settings.embedding_provider == "ollama":
            import httpx

            timeout = min(self.settings.request_timeout, 30.0)
            response = httpx.post(
                f"{self.settings.ollama_base_url}/api/embed",
                json={"model": self.settings.ollama_embed_model, "input": texts},
                timeout=timeout,
            )
            if response.status_code in (404, 405):
                # Older Ollama versions expose /api/embeddings and accept one prompt at a time.
                vectors = []
                for text in texts:
                    legacy = httpx.post(
                        f"{self.settings.ollama_base_url}/api/embeddings",
                        json={"model": self.settings.ollama_embed_model, "prompt": text},
                        timeout=timeout,
                    )
                    legacy.raise_for_status()
                    vector = legacy.json().get("embedding")
                    if not vector:
                        raise RuntimeError("Ollama returned an invalid legacy embedding response.")
                    vectors.append(vector)
                return vectors
            response.raise_for_status()
            vectors = response.json().get("embeddings")
            if not vectors or len(vectors) != len(texts):
                raise RuntimeError("Ollama returned an invalid embedding response.")
            return vectors
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Sentence Transformers is selected but optional package sentence-transformers is not installed."
            ) from exc
        if self._model is None:
            self._model = SentenceTransformer(self.settings.sentence_transformer_model)
        return self._model.encode(texts, normalize_embeddings=True).tolist()
