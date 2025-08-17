#!/usr/bin/env bash
# resource_monitor.sh — simple monitoring script
#
# This helper monitors GPU and CPU usage and prints statistics at
# regular intervals.  It can be run inside the container or on the
# host.  When executed inside a Runpod pod, it uses `nvidia-smi` to
# display GPU memory utilisation and the top CPU processes.  This
# script is optional but can help you tune batch sizes and detect
# memory leaks during experiments【639858406126874†L220-L230】.

set -euo pipefail

INTERVAL=${INTERVAL:-5}

echo "Resource monitor started.  Press Ctrl+C to exit."

while true; do
    echo "\n----- $(date) -----"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory --format=csv,nounits,noheader | \
            awk -F',' '{printf("GPU %s (%s): total %s MiB, used %s MiB, free %s MiB, gpu util %s%%, mem util %s%%\n", $1, $2, $3, $4, $5, $6, $7)}'
    fi
    # Show top 5 CPU processes by memory usage
    ps -eo pid,comm,%mem,%cpu --sort=-%mem | head -n 6
    sleep "$INTERVAL"
done