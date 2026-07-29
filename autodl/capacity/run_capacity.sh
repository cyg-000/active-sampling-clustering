#!/usr/bin/env bash
#
#
cd "$(dirname "$0")"
JOBS=${JOBS:-4}
LOGDIR=logs; mkdir -p $LOGDIR

python -c "import torch,sys; ok=torch.cuda.is_available(); print('CUDA:',ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)" \
  || { echo '!! No CUDA'; exit 1; }
[ -f dataset.npz ] || { echo '!! Missing dataset.npz'; exit 1; }

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 10 > $LOGDIR/gpu_cap.csv 2>/dev/null &
SMI=$!; trap 'kill $SMI 2>/dev/null' EXIT

run() { local tag=$1; shift
  echo ">> $tag"
  python train.py --tag "$tag" "$@" > "$LOGDIR/$tag.log" 2>&1 || echo "$tag" >> "$LOGDIR/FAILED_cap"; }

QUEUE=()
for S in 0 1 2; do
  for L in 0 0.5 1 2 4; do
    tag="CAP${L}_s$S"
    QUEUE+=("$tag --pairing meanmax --capacity $L --cap 4 --seed $S")
  done
done

echo "${#QUEUE[@]} models, $JOBS concurrent"
for job in "${QUEUE[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
  run $job &
done
wait
kill $SMI 2>/dev/null

echo; echo "===== Status ====="
for dir in runs/CAP*/; do
  t=$(basename "$dir")
  [ -f "$dir/status.json" ] && python -c "import json,sys;s=json.load(open(sys.argv[1]));print('  %-14s %-8s %s'%(sys.argv[2],s.get('status'),{k:(round(v,4) if isinstance(v,float) else v) for k,v in s.items() if k!='status'}))" "$dir/status.json" "$t" \
    || echo "  $t <no status.json>"
done
[ -f "$LOGDIR/FAILED_cap" ] && { echo; echo "!! FAILED:"; cat "$LOGDIR/FAILED_cap"; }

echo; echo "===== Ablation table (slope_k_vs_n vs lambda) ====="
python evaluate.py --tags $(for S in 0 1 2; do for L in 0 0.5 1 2 4; do echo -n "CAP${L}_s$S "; done; done) 2>&1 | tee $LOGDIR/evaluate_cap.log
echo
echo ">> See README capacity section: check whether slope_k moves toward 0.061 with lambda,"
echo "   whether slope_num decreases (dissociation), and ARI does not collapse."
