"""Summarize same-shape eager/replay JSONL diagnostics without torch."""
import argparse
import json
import math
from pathlib import Path


def summarize(directory):
    comparisons, errors, installed = [], [], set()
    for path in sorted(directory.glob("*.jsonl")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            try:
                event = json.loads(line)
                kind = event["event"]
                if kind == "installed":
                    installed.add(event["pid"])
                elif kind == "error":
                    errors.append(event)
                elif kind == "comparison":
                    if (event["actual_bs"] <= 0 or event["numel"] <= 0
                            or not 0 <= event["argmax_agreement"] <= event["actual_bs"]
                            or type(event["bitwise_equal"]) is not bool
                            or any(not math.isfinite(event[k]) or event[k] < 0
                                   for k in ("max_abs", "rms"))):
                        raise ValueError("Invalid comparison metrics")
                    comparisons.append(event)
                else:
                    raise ValueError(f"Unknown event {kind}")
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error
    pids = {e["pid"] for e in comparisons}
    if not comparisons:
        errors.append({"event": "error", "error": "No replay comparisons recorded"})
    if installed != pids:
        errors.append({"event": "error", "error": "Installed/comparing worker PIDs differ"})
    elements = sum(e["numel"] for e in comparisons)
    return {"comparisons": len(comparisons), "worker_pids": sorted(pids),
            "rows": sum(e["actual_bs"] for e in comparisons),
            "argmax_disagreements": sum(e["actual_bs"] - e["argmax_agreement"] for e in comparisons),
            "bitwise_equal_replays": sum(e["bitwise_equal"] for e in comparisons),
            "max_abs": max((e["max_abs"] for e in comparisons), default=None),
            "rms": (math.sqrt(sum(e["rms"] ** 2 * e["numel"] for e in comparisons) / elements)
                    if elements else None),
            "errors": errors,
            "note": "All TP workers included; rows are worker observations, not unique requests."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = summarize(args.output / "replay-control")
    (args.output / "replay-control-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
