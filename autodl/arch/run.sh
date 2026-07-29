#!/usr/bin/env bash
#
cd "$(dirname "$0")"
JOBS=${JOBS:-4}
LOGDIR=logs; mkdir -p $LOGDIR

python -c "import torch,sys; ok=torch.cuda.is_available(); print('CUDA:',ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)" \
  || { echo '!! No CUDA'; exit 1; }
[ -f dataset.npz ] || { echo '!! Missing dataset.npz'; exit 1; }

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 10 > $LOGDIR/gpu.csv 2>/dev/null &
SMI=$!; trap 'kill $SMI 2>/dev/null' EXIT

run() { local tag=$1; shift
  echo ">> $tag"
  python train.py --tag "$tag" "$@" > "$LOGDIR/$tag.log" 2>&1 || echo "$tag" >> "$LOGDIR/FAILED"; }

QUEUE=()
for S in 0 1 2; do
  QUEUE+=("A_base_s$S        --pairing meanmax --seed $S")
  QUEUE+=("A_ncount_s$S      --pairing meanmax --ncount --seed $S")
  QUEUE+=("A_dpool_s$S       --pairing dpool   --seed $S")
  QUEUE+=("A_dpool_nc_s$S    --pairing dpool   --ncount --seed $S")
  QUEUE+=("A_attn_s$S        --pairing attn    --seed $S")
  QUEUE+=("A_attn_nc_s$S     --pairing attn    --ncount --seed $S")
done

echo "${#QUEUE[@]} models, $JOBS concurrent"
for job in "${QUEUE[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
  run $job &
done
wait
kill $SMI 2>/dev/null

echo; echo "===== Status ====="
for dir in runs/*/; do
  t=$(basename "$dir")
  [ -f "$dir/status.json" ] && python -c "import json,sys;s=json.load(open(sys.argv[1]));print('  %-16s %-8s %s'%(sys.argv[2],s.get('status'),{k:(round(v,4) if isinstance(v,float) else v) for k,v in s.items() if k!='status'}))" "$dir/status.json" "$t" \
    || echo "  $t <no status.json>"
done
[ -f "$LOGDIR/FAILED" ] && { echo; echo "!! FAILED:"; cat "$LOGDIR/FAILED"; }

echo; awk -F, 'NR>1{gsub(/ %/,"",$2); s+=$2;n++; if($2+0>m)m=$2} END{if(n)printf "GPU avg %.0f%%  peak %.0f%%\n", s/n, m}' $LOGDIR/gpu.csv 2>/dev/null

echo; echo "===== Ablation table (see first column slope_k_vs_n) ====="
python evaluate.py 2>&1 | tee $LOGDIR/evaluate.log

echo; echo "===== Hidden-layer probe ====="
python probe.py --tags A_base_s0 A_dpool_nc_s0 A_attn_nc_s0 2>&1 | tee $LOGDIR/probe.log
