#!/usr/bin/env bash
# Train the grad_potential arm of sym_jg_full on CPU, reusing the CPU-trained
# incumbent (l-ft-k-svd-clamp) from runs/kappa_full_cpu.  See g-potential plan.
set -uo pipefail
cd ~/Documents/github/FTNODE

LOG=scripts/g_potential.log
: > "$LOG"
echo "[$(date '+%F %T')] START grad_potential run" | tee -a "$LOG"

# 1. Reuse the CPU-trained incumbent.  Copy checkpoints AND histories: --skip-existing
#    gates on ckpt AND hist existing.  Source is kappa_full_cpu (not GPU kappa_full) so
#    both arms of the comparison are CPU-produced.
mkdir -p runs/sym_jg_full/variants
cp -r runs/kappa_full_cpu/variants/l-ft-k-svd-clamp runs/sym_jg_full/variants/
n_pth=$(ls runs/sym_jg_full/variants/l-ft-k-svd-clamp/*.pth | wc -l)
n_hist=$(ls runs/sym_jg_full/variants/l-ft-k-svd-clamp/*.hist.json | wc -l)
echo "[$(date '+%F %T')] copied incumbent: $n_pth ckpts, $n_hist histories (expect 10/10)" | tee -a "$LOG"

# 2. Train ONLY grad_potential, one single-threaded process per seed, --no-rollouts,
#    staggered startup, resumable.
echo "[$(date '+%F %T')] launching 10 grad_potential shards" | tee -a "$LOG"
for s in $(seq 0 9); do
  OMP_NUM_THREADS=1 uv run ftnode-train experiments/duffing/sym_jg_full.yaml \
    --only svd_clamp+grad_potential --seeds "$s" --device cpu \
    --no-rollouts --skip-existing --quiet \
    >> "scripts/g_potential.seed${s}.log" 2>&1 &
  sleep 1
done
wait
echo "[$(date '+%F %T')] all shards finished" | tee -a "$LOG"

# 3. Final single-process pass builds rollouts.npz for BOTH arms.
echo "[$(date '+%F %T')] building rollout cache (both arms)" | tee -a "$LOG"
uv run ftnode-train experiments/duffing/sym_jg_full.yaml \
  --device cpu --skip-existing >> "$LOG" 2>&1
echo "[$(date '+%F %T')] DONE rc=$?" | tee -a "$LOG"

# quick sanity summary
echo "[$(date '+%F %T')] grad_potential ckpts: $(ls runs/sym_jg_full/variants/svd_clamp+grad_potential/*.pth 2>/dev/null | wc -l)/10" | tee -a "$LOG"
echo "[$(date '+%F %T')] rollouts.npz: $(ls -la runs/sym_jg_full/rollouts.npz 2>/dev/null || echo MISSING)" | tee -a "$LOG"
