"""Run with python -m scripts.validate_dataset [path ...]."""

import argparse
from collections import Counter

from framework.schema import load_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Validate benchmark JSONL and show category coverage"
    )
    parser.add_argument(
        "paths", nargs="*", default=["datasets/benchmark.jsonl", "datasets/adversarial.jsonl"]
    )
    args = parser.parse_args()
    for path in args.paths:
        cases = load_dataset(path)
        print(f"{path}: {len(cases)} valid cases")
        for category, count in sorted(Counter(case.category for case in cases).items()):
            print(f"  {category}: {count}")


if __name__ == "__main__":
    main()
