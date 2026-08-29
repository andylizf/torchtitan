#!/usr/bin/env bash
# Restart the della-side evolution loop, preserving the environment it needs.
#
# The loop's credentials and knobs live only in its own environment -- there is
# no env file it reads at startup -- so a restart has to carry them across from
# the process being replaced. Word-splitting a saved environment does not
# survive contact with it: SSH_CONNECTION and LESSOPEN hold spaces, and
# `env $(cat saved)` turns the second word of the first such value into the
# command name. Read it line by line instead, value verbatim to end of line.
set -uo pipefail
ROOT=/scratch/gpfs/TRIDAO/al9080/terminal-rl
VENV=/scratch/gpfs/TRIDAO/al9080/titan-rl/bin/python
LOG=$ROOT/logs/evolve_ondella.log
ENVFILE=${1:-$ROOT/tmp/evolve.env}
WORKERS=${EVOLVE_WORKERS:-16}
INTERVAL=${EVOLVE_INTERVAL:-120}

[ -f "$ENVFILE" ] || { echo "no env snapshot at $ENVFILE"; exit 1; }

if pgrep -f evolve_ondella.py >/dev/null; then
    echo "stopping the running loop"
    pkill -TERM -f evolve_ondella.py
    for _ in $(seq 1 20); do
        pgrep -f evolve_ondella.py >/dev/null || break
        sleep 3
    done
    pgrep -f evolve_ondella.py >/dev/null && pkill -9 -f evolve_ondella.py
    sleep 2
fi

# Session-scoped variables belong to whoever was logged in when the snapshot was
# taken, not to the loop; carrying them forward is at best noise.
while IFS= read -r line; do
    case "$line" in
        ""|"#"*) continue ;;
        SSH_*|_=*|SHLVL=*|PWD=*|OLDPWD=*|TERM=*|LESSOPEN=*|LESSCLOSE=*) continue ;;
        # Exported shell functions arrive as `BASH_FUNC_name%%=() {` and run
        # over several lines, so every line after the first parses as garbage.
        # They are the shell's, not the loop's.
        BASH_FUNC_*|"}"|"("*|" "*|")"*) continue ;;
    esac
    key=${line%%=*}
    case "$key" in
        [A-Za-z_][A-Za-z0-9_]*) export "$key=${line#*=}" ;;
    esac
done < "$ENVFILE"

cd "$ROOT" || exit 1
setsid nohup "$VENV" evolve-onhost/scripts/evolve_ondella.py \
    --interval "$INTERVAL" --workers "$WORKERS" >> "$LOG" 2>&1 < /dev/null &
sleep 15

PID=$(pgrep -f evolve_ondella.py | head -1)
if [ -z "$PID" ]; then
    echo "FAILED to start; last lines of $LOG:"
    tail -5 "$LOG"
    exit 1
fi
echo "running pid=$PID workers=$WORKERS interval=$INTERVAL"
echo "  SWE_RETUNE_AGENT=$(tr '\0' '\n' < /proc/$PID/environ | sed -n 's/^SWE_RETUNE_AGENT=//p')"
echo "  DAYTONA_API_KEY set: $(tr '\0' '\n' < /proc/$PID/environ | grep -c '^DAYTONA_API_KEY=')"
echo "  SYNTH_ENV_FILE=$(tr '\0' '\n' < /proc/$PID/environ | sed -n 's/^SYNTH_ENV_FILE=//p')"
