#!/usr/bin/env bash
#
cd "$(dirname "$0")"
JOBS=${JOBS:-4}
LOGDIR=logs; mkdir -p $LOGDIR
python -c "import torch,sys; ok=torch.cuda.is_available(); print('CUDA:',ok); sys.exit(0 if ok else 1)" || { echo '!! No CUDA'; exit 1; }
[ -f dataset.npz ] || { echo '!! Missing dataset.npz'; exit 1; }
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 10 > $LOGDIR/gpu_cap2.csv 2>/dev/null &
SMI=$!; trap 'kill $SMI 2>/dev/null' EXIT
run() { local tag=$1; shift; echo ">> $tag"
  python train.py --tag "$tag" "$@" > "$LOGDIR/$tag.log" 2>&1 || echo "$tag" >> "$LOGDIR/FAILED_cap2"; }

QUEUE=()
for S in 0 1 2; do
  for L in 0.1 0.15 0.2 0.3; do
    QUEUE+=("CAP${L}_s$S --pairing meanmax --capacity $L --cap 4 --seed $S")
  done
done
echo "${#QUEUE[@]} models, $JOBS concurrent"
for job in "${QUEUE[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
  run $job &
done
wait; kill $SMI 2>/dev/null

echo; echo "===== Low-lambda ablation (slope_k / slope_num / ARI) ====="
echo "     (round 1 lambda in {0,0.5,1,2,4} at ../capacity/runs/comparison.csv)"
python evaluate.py --tags $(for S in 0 1 2; do for L in 0.1 0.15 0.2 0.3; do echo -n "CAP${L}_s$S "; done; done) 2>&1 | tee $LOGDIR/evaluate_cap2.log
