#!/usr/bin/env bash
# bench_pinned.sh — usage: ./bench_pinned.sh "0,2,4,6" python bench.py
# Set the frequency state first (once per session, as root): see bench_mode.sh
set -euo pipefail
CPUS="$1"; shift

# accept both "0,2,4" and "0-7" forms
N=$(python3 -c "
s='$CPUS'; c=0
for p in s.split(','):
    a,_,b = p.partition('-'); c += int(b)-int(a)+1 if b else 1
print(c)")

export RAYON_NUM_THREADS=$N OMP_NUM_THREADS=$N MKL_NUM_THREADS=$N \
       OPENBLAS_NUM_THREADS=$N NUMEXPR_NUM_THREADS=$N NUMBA_NUM_THREADS=$N

TURBO=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo n/a)
GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)
[[ $TURBO == 1 && $GOV == performance ]] ||
    echo "bench_pinned: WARNING turbo=$TURBO governor=$GOV — run bench_mode.sh on" >&2

{ echo "=== $(date -Is) cpus=$CPUS n=$N turbo=$TURBO governor=$GOV"
  lscpu | grep -E 'Model name|MHz'; } >> bench_env.log

exec numactl --membind=0 taskset -c "$CPUS" "$@"
