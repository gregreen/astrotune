#!/usr/bin/env bash
#
# Three experiments on one question: have the per-group degrees of freedom reached
# their optimum, or has the network stopped at 20 epochs while nu was still moving?
#
# All start from the iteration-2 run and change only the network fit:
#
#   e64         --epochs 64, everything else as in models/latent_student_t_iter2_20cat
#   e64_nu14_6  the same, with the prior on raw moved from 6 +- 2 to 14 +- 6
#   e64_nu30_16 the same, with the prior on raw moved to 30 +- 16
#
# nu = 2 + softplus(raw), so raw = 14 +- 6 puts the prior mean at nu = 16.0 and
# raw = 30 +- 16 at nu = 32.0 (against 8.0 before). Movement in the prior is close to
# one-to-one there, so the prior is essentially that spread in nu. The runs separate
# "the fit needs more steps" (e64) from "the prior is holding nu down" (the other
# two), and the last of them asks how far nu goes once the prior has all but let go.
#
# The star table is copied rather than rebuilt: it is a function of the prepared
# tables and the bijector only, so the two runs then see bit-identical stars and a
# difference between them is a difference in the fit alone.
#
# Usage: scripts/nu_experiments.sh          (run in the background, ~40 min each)
#
set -uo pipefail
cd "$(dirname "$0")/.."

RUN=models/latent_student_t_iter2_20cat
BASE=$RUN/delta
PY=.venv/bin/python

for tool in "$PY" .venv/bin/astrotune-delta-train "$BASE/star_table.npz"; do
    [ -e "$tool" ] || { echo "missing $tool" >&2; exit 1; }
done

experiment() {
    local name=$1 out=$2
    shift 2
    # Idempotent: a finished experiment is skipped, so this script can be re-run to
    # add one more arm without repeating the earlier ones.
    if [ -f "$out/evaluation/comparison.json" ]; then
        echo "=== ${name} already done, skipping"
        return 0
    fi
    mkdir -p "$out"
    # A copy, not a link: save_star_table truncates in place, so a --rebuild in an
    # experiment would otherwise take the base table with it.
    [ -f "$out/star_table.npz" ] || cp "$BASE/star_table.npz" "$out/star_table.npz"
    local log=logs/nu_exp_${name}.log
    echo "=== ${name} starting $(date)"
    "$PY" .venv/bin/astrotune-delta-train --run-dir "$RUN" --out-dir "$out" "$@" \
        > "$log" 2>&1
    local rc=$?
    echo "=== ${name} exit=${rc} at $(date) (log ${log})"
    return $rc
}

experiment e64 "$RUN/delta_e64" --epochs 64
experiment e64_nu14_6 "$RUN/delta_e64_nu14_6" --epochs 64 \
    --nu-prior-mean 14 --nu-prior-sd 6
experiment e64_nu30_16 "$RUN/delta_e64_nu30_16" --epochs 64 \
    --nu-prior-mean 30 --nu-prior-sd 16

echo "=== all done $(date)"
