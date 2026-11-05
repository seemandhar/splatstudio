---
name: paper-todo
description: List the next papers to reproduce from the 100-paper plan, ranked by priority and difficulty, with their current reproduction status in reproductions/.
argument-hint: [n=10] [--area Densification] [--all]
---

# /paper-todo

Arguments: `$ARGUMENTS`

Run:
```bash
cd /home/sejain/splatify && python3 - <<'EOF'
import json, pathlib
papers = json.load(open("100-paper-plan/papers_100.json"))["papers"]
done = {}
for state in pathlib.Path("reproductions").glob("*/state.json"):
    s = json.loads(state.read_text())
    done[s.get("plan_id")] = (s.get("slug"), s.get("outcome") or next((k for k, v in reversed(list(s.get("stages", {}).items())) if v == "done"), "started"))
todo = [p for p in papers if str(p.get("status", "")).startswith("Proposed")]
todo.sort(key=lambda p: (p.get("difficulty") or 9, p.get("priority") or 9, p["id"]))
for p in todo:
    st = done.get(p["id"])
    print(f'#{p["id"]:>3} d{p.get("difficulty")} p{p.get("priority")} {p.get("area",""):<16} {p["full_title"][:80]}'
          + (f'  [{st[0]}: {st[1]}]' if st else "")
          + f'\n      route: {p.get("integration_route","")} | code: {p.get("code_url") or "—"}')
EOF
```
Show the first n (default 10) unless `--all`; filter by `--area` if given. Recommend the next
one with a one-line reason (lowest difficulty whose route is a native loss/density component
and that reports Mip-NeRF 360 numbers), and the exact command: `/reproduce plan#<id>`.
