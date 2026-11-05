#!/usr/bin/env python3
"""Write reproductions/PROGRESS.md: one row per plan paper with reproduction status and numbers."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
papers = json.loads((ROOT / "100-paper-plan" / "papers_100.json").read_text())["papers"]
states = {}
for path in (ROOT / "reproductions").glob("*/state.json"):
    state = json.loads(path.read_text())
    if state.get("plan_id") is not None:
        states[state["plan_id"]] = (path.parent, state)

rows = ["| # | paper | stage | outcome | ours vs paper (PSNR) |", "|---:|---|---|---|---|"]
counts = {}
for p in sorted(papers, key=lambda p: p["id"]):
    entry = states.get(p["id"])
    if entry is None:
        rows.append(f"| {p['id']} | {p['full_title'][:70]} | — | — | |")
        continue
    workspace, state = entry
    stages = [k for k, v in state.get("stages", {}).items() if str(v).startswith("done")]
    outcome = state.get("outcome") or "in progress"
    counts[outcome] = counts.get(outcome, 0) + 1
    rows.append(f"| {p['id']} | [{p['full_title'][:70]}]({workspace.name}/report.md) | "
                f"{stages[-1] if stages else 'started'} | {outcome} | {state.get('headline', '')} |")
summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "none started"
text = f"# Reproduction progress\n\n{len(states)}/{len(papers)} started: {summary}\n\n" + "\n".join(rows) + "\n"
(ROOT / "reproductions" / "PROGRESS.md").write_text(text)
print(text[:2000])
