#!/usr/bin/env bash
# bench_mode.sh — usage: sudo ./bench_mode.sh on|off
# Fixes the CPU clock for a benchmarking session so short and long runs see the
# same frequency. Run once before a campaign and once after, not per invocation.
set -euo pipefail
case "${1:-}" in
    on)  echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo
         cpupower frequency-set -g performance >/dev/null ;;
    off) echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo
         cpupower frequency-set -g powersave >/dev/null ;;
    *)   echo "usage: sudo $0 on|off" >&2; exit 2 ;;
esac
echo "turbo_off=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)" \
     "governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
