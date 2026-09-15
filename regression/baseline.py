import argparse
from pathlib import Path

from framework.storage import read_results, write_csv
from regression.compare_runs import validate_complete


def main():
    parser = argparse.ArgumentParser(description="Explicitly promote a completed run to baseline")
    parser.add_argument("source")
    parser.add_argument("--output", default="results/baseline_results.csv")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    try:
        frame = read_results(args.source)
        validate_complete(frame)
        if (frame.status != "ok").any():
            raise ValueError("Resolve generation/evaluation errors before promoting a baseline")
        if "dataset_complete" in frame and not frame.dataset_complete.all():
            raise ValueError("A smoke-test subset cannot be promoted as the release baseline")
        for column in ("judge_error", "semantic_error"):
            if column in frame and frame[column].fillna("").ne("").any():
                raise ValueError(f"Resolve {column} before promoting a baseline")
        if Path(args.output).exists() and not args.replace:
            raise ValueError("Baseline exists; pass --replace to replace it")
        write_csv(frame, args.output)
        print(f"Baseline saved: {args.output}")
        return 0
    except (ValueError, OSError) as exc:
        print(f"Baseline rejected: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
