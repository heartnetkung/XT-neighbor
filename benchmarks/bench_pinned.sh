#!/usr/bin/env bash
# bench_pinned.sh — usage: ./bench_pinned.sh "0,2,4,6" python bench.py
set -euo pipefail
CPUS="$1"; shift
N=$(python3 -c "s='$CPUS'; print(len(s.split(',')))")

export RAYON_NUM_THREADS=$N OMP_NUM_THREADS=$N MKL_NUM_THREADS=$N \
       OPENBLAS_NUM_THREADS=$N NUMEXPR_NUM_THREADS=$N NUMBA_NUM_THREADS=$N \
       OMP_PROC_BIND=close OMP_PLACES=cores

{ echo "cpus=$CPUS n=$N"; lscpu | grep -E 'Model name|MHz'; 
  cat /sys/devices/system/cpu/intel_pstate/no_turbo; } >> bench_env.log

exec numactl --membind=0 taskset -c "$CPUS" "$@"
