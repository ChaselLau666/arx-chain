#!/usr/bin/env bash
# Profile the Human DAgger control process with py-spy and write results to a log.
#
# Usage: run this any time after starting 05_/20_; it waits for the frontend,
# picks the control child (the CPU-bound one), samples it, and writes:
#   /tmp/dagger_profile/<timestamp>/pyspy_top.txt    - function hotspots (2x30s)
#   /tmp/dagger_profile/<timestamp>/flame.svg        - flamegraph
#   /tmp/dagger_profile/<timestamp>/context.txt      - top/thread snapshot
# Sampling is read-only (ptrace attach); the py-spy binary carries
# cap_sys_ptrace (set once via setcap), so no sudo is needed.
set -Eeuo pipefail

PYSPY=/home/arx/miniconda3/envs/act/bin/py-spy

out_root=/tmp/dagger_profile/$(date +%Y%m%dT%H%M%S)
mkdir -p "$out_root"

echo "Waiting for a human_dagger.py control process (Ctrl-C to abort)..."
control_pid=""
until [[ -n "$control_pid" ]]; do
  # The supervisor forks control/policy/recorder; the control process is the
  # busiest human_dagger.py PID. Pick the top-CPU one after it has warmed up.
  control_pid=$(ps -eo pid,pcpu,args --sort=-pcpu \
    | awk '/[h]uman_dagger\.py/ {print $1; exit}')
  [[ -n "$control_pid" ]] || sleep 2
done
sleep 5  # let it settle past startup

echo "Profiling PID ${control_pid}; results in ${out_root}"
{
  echo "=== $(date) PID ${control_pid} ==="
  ps -o pid,ppid,pcpu,pmem,args -p "$control_pid"
  echo "=== all human_dagger pids ==="
  ps -eo pid,ppid,pcpu,args | grep '[h]uman_dagger.py'
  echo "=== threads ==="
  top -bn1 -H -p "$control_pid" | head -20
} > "$out_root/context.txt"

# Two 30s top passes: catch both POLICY and HUMAN phases if the operator is
# switching; each pass writes a self-contained hotspot table.
for pass in 1 2; do
  echo "py-spy top pass ${pass}/2 (30 s)..."
  {
    echo "=== pass ${pass} $(date) ==="
    "$PYSPY" top --pid "$control_pid" --duration 30 2>&1
  } >> "$out_root/pyspy_top.txt"
done

echo "py-spy record (30 s flamegraph)..."
"$PYSPY" record --pid "$control_pid" \
  -o "$out_root/flame.svg" -d 30 2>>"$out_root/pyspy_top.txt" || true

echo
echo "Done. Read with:"
echo "  less ${out_root}/pyspy_top.txt"
echo "  (flamegraph: scp ${out_root}/flame.svg to a browser)"
