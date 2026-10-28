"""Add/remove temporary replay coverage logging. Not for timing runs."""
import argparse
from pathlib import Path

START = "        # BEGIN LAPS_VERIFY instrumentation\n"
END = "        # END LAPS_VERIFY instrumentation\n"
BLOCK = '''        # BEGIN LAPS_VERIFY instrumentation
        import json as _verify_json
        print("LAPS_VERIFY " + _verify_json.dumps({
            "actual_bs": actual_bs,
            "target_bs": target_bs,
            "width": target_seq_len,
            "extend": extend_lens_cpu,
            "kv": forward_batch.seq_lens[:actual_bs].cpu().tolist(),
        }), flush=True)
        # END LAPS_VERIFY instrumentation
'''


def instrument(source, remove=False):
    if START in source:
        start = source.index(START)
        end = source.index(END, start) + len(END)
        source = source[:start] + source[end:]
    if remove:
        return source
    needle = "        output = self.batch_prefill_outputs[(target_bs, target_seq_len)]\n"
    if source.count(needle) != 1:
        raise ValueError("Runner differs from supplied source; refusing to instrument")
    result = source.replace(needle, BLOCK + needle)
    compile(result, "piecewise_cuda_graph_runner.py", "exec")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remove", action="store_true")
    parser.add_argument("--upstream", default="vendor/LAPS")
    args = parser.parse_args()
    path = Path(args.upstream) / "python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py"
    path.write_text(instrument(path.read_text(), args.remove))
    print("Replay coverage logging " + ("removed" if args.remove else "installed"))
