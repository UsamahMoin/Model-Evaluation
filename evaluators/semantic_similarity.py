from functools import lru_cache

from framework.schema import Evaluation


@lru_cache(maxsize=2)
def load_encoder(model: str, local_files_only: bool = True):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install requirements-semantic.txt to enable semantic scoring") from exc
    return SentenceTransformer(
        model, device="cpu", local_files_only=local_files_only, trust_remote_code=False
    )


def evaluate(
    response: str,
    expected: str | list[str],
    *,
    model: str,
    threshold: float = 0.8,
    local_files_only: bool = True,
) -> Evaluation:
    references = [expected] if isinstance(expected, str) else expected
    encoder = load_encoder(model, local_files_only)
    vectors = encoder.encode([response, *references], normalize_embeddings=True)
    score = max(0.0, min(1.0, float((vectors[1:] @ vectors[0]).max())))
    return Evaluation(
        passed=score >= threshold,
        score=score,
        reason=f"Cosine similarity {score:.3f}; threshold {threshold:.3f}. Similarity is a signal, not a truth judgment.",
    )
