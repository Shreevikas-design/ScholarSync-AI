from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path

from config import Settings

_MODEL_LOCK = threading.RLock()


@lru_cache(maxsize=2)
def _onnx_embedder(cache_dir: str):
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

    model_cache = Path(cache_dir) / "all-MiniLM-L6-v2"

    class ScholarSyncMiniLM(ONNXMiniLM_L6_V2):
        DOWNLOAD_PATH = model_cache

    return ScholarSyncMiniLM()


@lru_cache(maxsize=2)
def _sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class EmbeddingService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.settings.embedding_provider == "chroma-default":
            try:
                model = _onnx_embedder(str(self.settings.data_dir / "models" / "embeddings"))
                with _MODEL_LOCK:
                    vectors = model(texts)
                return [vector.tolist() if hasattr(vector, "tolist") else list(vector) for vector in vectors]
            except Exception as exc:
                raise RuntimeError(
                    f"Embedded Chroma ONNX embeddings unavailable ({type(exc).__name__}). "
                    "Keyword retrieval remains available; check model-download access and onnxruntime."
                ) from exc
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
            model = _sentence_transformer(self.settings.sentence_transformer_model)
        except ImportError as exc:
            raise RuntimeError(
                "Sentence Transformers is selected but optional package sentence-transformers is not installed."
            ) from exc
        if self._model is None:
            self._model = model
        with _MODEL_LOCK:
            vectors = self._model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()
