#!/usr/bin/env bash
# ============================================================================
#
#
#
# ============================================================================
cd "$(dirname "$0")"
JOBS=${JOBS:-4}
SEEDS=${SEEDS:-"0 1 2 3 4"}                         # 5 seeds -> tighter CI on frontier (needed for test)
LAMBDAS=${LAMBDAS:-"0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.10"}
LOGDIR=logs; mkdir -p $LOGDIR

python -c "import torch,sys; ok=torch.cuda.is_available(); print('CUDA:',ok); sys.exit(0 if ok else 1)" \
  || { echo '!! No CUDA'; exit 1; }
[ -f dataset.npz ] || { echo '!! Missing dataset.npz'; exit 1; }
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 10 \
  > $LOGDIR/gpu_dense.csv 2>/dev/null & SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT

run() { local tag=$1; shift; echo ">> $tag"
  python train.py --tag "$tag" "$@" > "$LOGDIR/$tag.log" 2>&1 \
    || echo "$tag" >> "$LOGDIR/FAILED_dense"; }

QUEUE=()
for S in $SEEDS; do
  for L in $LAMBDAS; do
    QUEUE+=("CAP${L}_s$S --pairing meanmax --capacity $L --cap 4 --seed $S")
  done
done
echo "${#QUEUE[@]} models (lambda x seeds), $JOBS concurrent -- ~1 hour estimated (5090)"
for job in "${QUEUE[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
  run $job &
done
wait; kill $SMI 2>/dev/null

echo; echo "===== Evaluate (same test set) -> runs/comparison.csv ====="
TAGS=""
for S in $SEEDS; do for L in $LAMBDAS; do TAGS+="CAP${L}_s$S "; done; done
python evaluate.py --tags $TAGS 2>&1 | tee $LOGDIR/evaluate_dense.log

echo; echo "===== Frontier analysis + pre-registered positive-difference test ====="
python analyze_frontier.py 2>&1 | tee $LOGDIR/analyze_dense.log
echo
echo "Send back runs/comparison.csv and outputs/frontier_verdict.txt."
