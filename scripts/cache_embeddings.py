"""One-time explicit download; subsequent semantic evaluations use cached files."""

from framework.schema import load_config


def main():
    from sentence_transformers import SentenceTransformer

    config = load_config("config.yaml")
    SentenceTransformer(config.semantic.model, device="cpu", trust_remote_code=False)
    print(f"Cached {config.semantic.model}. Semantic evaluation can now run offline.")


if __name__ == "__main__":
    main()
