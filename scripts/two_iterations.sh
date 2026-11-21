#!/usr/bin/env bash
# Two iterations of (latent-y training -> delta model training), unattended.
#
# Iteration 1 fits the latent-y model with the quoted uncertainties and a
# Student's t with nu = 50 for every catalog -- the value at which the t and the
# old softened chi^2 agree to first order, so the run is a controlled comparison
# against latent_adam_lr1e4_20cat. Its delta model then calibrates the
# uncertainties and learns a nu per group.
#
# Iteration 2 rebuilds the observations with those calibrated uncertainties,
# which also carries that nu into the file, and retrains with the same
# hyperparameters. The trainer reads the per-catalog nu from the observations
# (see astrotune.train_latent.resolve_nu), so the second run's likelihood is the
# one its sigmas were calibrated under.
#
# The hyperparameters are latent_adam_lr1e4_20cat's: every field of its
# config.json that differs from the current defaults is passed explicitly.
#
#   scripts/two_iterations.sh                  # the whole chain
#   scripts/two_iterations.sh --from delta1    # resume after a failure
set -u
cd "$(dirname "$0")/.." || exit 1

FROM=latent1
if [ "${1:-}" = "--from" ]; then
  FROM=${2:?--from needs a step: latent1 apply1 delta1 obs2 latent2 apply2 delta2}
fi

PY=.venv/bin
RUN1=latent_student_t_lr1e4_20cat
RUN2=latent_student_t_iter2_20cat
OBS2=data/obs_student_t_iter2.h5

# The reference run's configuration.
COMMON=(
  --batch-size 65536
  --n-epochs 250
  --learning-rate 1e-4
  --pretrain-steps 256
  --freeze-vars embedding film
  --checkpoint-every-steps 128
)

_pending=0
step () {  # step <name> <log> <command...>
  local name=$1 log=$2
  shift 2
  if [ "$_pending" -eq 0 ]; then
    if [ "$name" != "$FROM" ]; then
      echo "--- skipping $name (resuming at $FROM)"
      return 0
    fi
    _pending=1
  fi
  echo "=== $name starting $(date '+%F %T')"
  "$@" > "$log" 2>&1
  local rc=$?
  echo "=== $name exit=$rc at $(date '+%F %T') (log $log)"
  if [ "$rc" -ne 0 ]; then
    echo "=== STOP: $name failed; nothing further will run"
    echo "=== resume with: scripts/two_iterations.sh --from $name"
    exit "$rc"
  fi
}

step latent1 logs/iter1_latent.log $PY/astrotune-train-latent \
  --run-name "$RUN1" \
  --observations-path data/observations_and_ctrl_points.h5 \
  "${COMMON[@]}" --data-term student_t --nu 50

# The delta model needs the harmonized estimates, which apply caches.
step apply1 logs/iter1_apply.log $PY/astrotune-apply --run-dir "models/$RUN1"

step delta1 logs/iter1_delta.log $PY/astrotune-delta-train --run-dir "models/$RUN1"

step obs2 logs/iter2_observations.log $PY/astrotune-observations \
  --sigma-source delta --delta-dir "models/$RUN1" -o "$OBS2"

step latent2 logs/iter2_latent.log $PY/astrotune-train-latent \
  --run-name "$RUN2" \
  --observations-path "$OBS2" \
  "${COMMON[@]}"

step apply2 logs/iter2_apply.log $PY/astrotune-apply --run-dir "models/$RUN2"

step delta2 logs/iter2_delta.log $PY/astrotune-delta-train --run-dir "models/$RUN2"

echo "=== all done $(date '+%F %T')"
