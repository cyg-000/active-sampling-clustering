#!/usr/bin/env bash
#
#
cd "$(dirname "$0")"
JOBS=${JOBS:-4}
LOGDIR=logs; mkdir -p $LOGDIR

python -c "import torch,sys; ok=torch.cuda.is_available(); print('CUDA:',ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)" \
  || { echo '!! No CUDA, fix environment first'; exit 1; }
[ -f dataset.npz ] || { echo '!! Missing dataset.npz; run build_dataset.py locally and upload'; exit 1; }

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
  --format=csv -l 10 > $LOGDIR/gpu.csv 2>/dev/null &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT

run() {   # run <tag> <args...>
  local tag=$1; shift
  echo ">> $tag"
  python train.py --tag "$tag" "$@" > "$LOGDIR/$tag.log" 2>&1 \
    || echo "$tag" >> "$LOGDIR/FAILED"
}

QUEUE=()
add() { QUEUE+=("$*"); }

for S in 0 1 2; do add "H_s$S --target human --seed $S"; done
add "R_shuffle --target shuffle --seed 0"    # control K, destroy geometry only
add "R_gmm     --target gmm     --seed 0"
add "R_dbscan  --target dbscan  --seed 0"
add "H_noscan --target human --scanpath none --seed 0"
add "H_nogru  --target human --no-gru        --seed 0"
for SUB in lasso voronoi full funnel; do add "H_$SUB --target human --subset $SUB --seed 0"; done

echo "${#QUEUE[@]} models, $JOBS concurrent"
for job in "${QUEUE[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
  run $job &
done
wait
kill $SMI 2>/dev/null

echo; echo "===== Summary ====="
for dir in runs/*/; do
  tag=$(basename "$dir")
  if [ -f "$dir/status.json" ]; then
    python -c "import json,sys; s=json.load(open(sys.argv[1])); print('  %-12s %-12s %s'%(sys.argv[2],s.get('status'),{k:v for k,v in s.items() if k!='status'}))" "$dir/status.json" "$tag"
  else
    echo "  $tag  <no status.json -- process may have been OOM-killed>"
  fi
done
[ -f "$LOGDIR/FAILED" ] && { echo; echo "!! Failed models (logs in $LOGDIR/):"; cat "$LOGDIR/FAILED"; }

echo; echo "GPU utilisation:"
awk -F, 'NR>1{gsub(/ %/,"",$2); s+=$2; n++; if($2+0>m)m=$2} END{if(n)printf "  avg %.0f%%  peak %.0f%%  (%d samples)\n", s/n, m, n}' $LOGDIR/gpu.csv 2>/dev/null

python evaluate.py 2>&1 | tee $LOGDIR/evaluate.log
echo "== See runs/comparison.csv =="
