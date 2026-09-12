#!/usr/bin/env bash
# Proof that sharded execution reproduces the single-process run exactly.
set -e
unset PYTHONPATH
PY=".venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8
T=tools/shardtest
COMMON="--n-train 4000 --n-val 1000 --epochs 40 --patience 5 \
 --select-epochs 5 --select-n 2000 --reps-candidates 1 3 --lr-candidates 0.05 0.01"

rm -rf $T/ref $T/s1 $T/s2 $T/abl $T/sel.json $T/merged.json $T/first.npz

echo "### reference: one process, seeds 0 1 2 3"
$PY -u -m src.run_study $COMMON --seeds 0 1 2 3 --ablation-seeds 1 \
    --out $T/ref > $T/ref.log 2>&1

echo "### selection only"
$PY -u -m src.run_study $COMMON --seeds 0 --selection-only $T/sel.json \
    --out $T/sel > $T/sel.log 2>&1

echo "### shard 1: seeds 0 1   (holds the ROC seed)"
$PY -u -m src.run_study $COMMON --seeds 0 1 --selection-from $T/sel.json \
    --roc-seed 0 --save-first $T/first.npz --no-ablation --threads 1 \
    --out $T/s1 > $T/s1.log 2>&1 &
echo "### shard 2: seeds 2 3"
$PY -u -m src.run_study $COMMON --seeds 2 3 --selection-from $T/sel.json \
    --roc-seed 0 --no-ablation --threads 1 \
    --out $T/s2 > $T/s2.log 2>&1 &
echo "### ablation alongside"
$PY -u -m src.run_study $COMMON --seeds 0 1 2 3 --selection-from $T/sel.json \
    --ablation-only --ablation-seeds 1 --threads 1 \
    --out $T/abl > $T/abl.log 2>&1 &
wait

echo "### merge"
$PY tools/merge_shards.py --shards $T/s1/metrics.json $T/s2/metrics.json \
    --selection $T/sel.json --ablation $T/abl/metrics.json \
    --out $T/merged.json

echo "### compare"
$PY tools/compare_metrics.py $T/ref/metrics.json $T/merged.json
