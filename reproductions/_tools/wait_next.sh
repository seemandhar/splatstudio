#!/usr/bin/env bash
# Block until some experiment not yet listed in the seen-file finishes; print it and record it.
# Usage: tools/wait_next.sh <seen-file> exp1 exp2 ...
seen="$1"; shift; touch "$seen"; cd "$(dirname "$0")/.."
while true; do
  for e in "$@"; do
    grep -qx "$e" "$seen" && continue
    f=$(python3 tools/bench.py status --exp "$e" 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print(int(all(j['status'] in('done','failed','cancelled') for j in d['jobs'])), json.dumps(d['counts']))" 2>/dev/null)
    case "$f" in 1*) echo "$e FINISHED ${f#1 }"; echo "$e" >> "$seen"; exit 0;; esac
  done
  sleep 120
done
