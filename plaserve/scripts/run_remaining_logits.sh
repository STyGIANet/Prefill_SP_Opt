#!/usr/bin/env bash
set -uo pipefail
KR=kernels/flashinfer-saturating-4fbcdb804781
OUT=results/remaining-logits-$(date +%Y%m%d-%H%M%S)
ARGS=(--kernel-root "$KR" --dump-logits)
ARGS+=(--cases cold-1,cold-6 --counts 1 8)
ARGS+=(--server-arg=--enable-fp32-lm-head)
python scripts/check_blackwell_model.py "${ARGS[@]}" --output "$OUT"
status=$?
echo "Checker exit status: $status; diagnostic output: $OUT"
if [[ -f "$OUT/eager-responses.json" && -f "$OUT/graphs-responses.json" ]]; then
    python scripts/summarize_model_logits.py "$OUT" || exit $?
fi
exit "$status"
