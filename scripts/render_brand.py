"""Generate the repository's original SVG artwork; no external assets required."""

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "docs/assets"
ROOT.mkdir(parents=True, exist_ok=True)


def text(x, y, value, size=18, color="#acb9ce", weight=400):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}">{value}</text>'


def save(name, width, height, body, background):
    source = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title">
<title id="title">SplatStudio: installable methods and traceable Gaussian splatting experiments</title>
<defs><radialGradient id="splat"><stop stop-color="#98ffe0" stop-opacity=".8"/><stop offset="1" stop-color="#5ed9bc" stop-opacity="0"/></radialGradient></defs>
<rect width="{width}" height="{height}" rx="24" fill="{background}"/>
<g font-family="Inter, DejaVu Sans, Arial, sans-serif">{body}</g></svg>'''
    (ROOT / name).write_text(source + "\n")


body = ""
for x in range(740, 1170, 34):
    for y in range(24, 440, 34):
        body += f'<circle cx="{x}" cy="{y}" r="1" fill="#293344"/>'
for i in range(96):
    t = i * 0.154
    x = 950 + 135 * math.sin(t)
    y = 218 + 148 * math.cos(t / 1.8)
    angle = i * 23
    color = ["#82e4c3", "#a5a0ff", "#ffaf8c"][i % 3]
    body += f'<ellipse cx="{x:.1f}" cy="{y:.1f}" rx="{14 + i % 17}" ry="{5 + i % 9}" transform="rotate({angle} {x:.1f} {y:.1f})" fill="{color}" fill-opacity=".14" stroke="{color}" stroke-opacity=".28"/>'
body += text(54, 60, "S P L A T S T U D I O", 16, "#8ce4ca", 700)
body += text(50, 150, "Your next method.", 53, "#f4f6fc", 700)
body += text(50, 215, "A shared foundation.", 53, "#f4f6fc", 700)
body += text(54, 264, "Create, install, train, and inspect Gaussian splatting methods.", 18)
body += text(54, 294, "Open components. Executable checks. Traceable results.", 18)
body += '<path d="M54 340 H685" stroke="#29364a"/>'
for x, number, label in [
    (54, "05", "NATIVE RECIPES"),
    (274, "33", "RESEARCH CONFIGS"),
    (513, "01", "PLUGIN SDK"),
]:
    body += text(x, 388, number, 33, "#f4f6fc", 600)
    body += text(x, 414, label, 11, "#8e9ab0", 600)
save("hero.svg", 1200, 460, body, "#0c1322")

body = text(40, 52, "A method is a package. An experiment leaves evidence.", 27, "#17243a", 700)
body += text(
    40,
    83,
    "The native SDK and the Nerfstudio research collection are separate, interoperable entry points.",
    15,
    "#596b80",
)
cards = [
    (
        40,
        "01  CREATE",
        "Your Python package",
        ["MethodSpec", "Renderer + density + losses", "Custom fields and probes"],
        "#e7e5ff",
    ),
    (
        340,
        "02  EXECUTE",
        "Native SplatStudio",
        [
            "3D / 2D gsplat rasterization",
            "Gradient-aware density control",
            "CUDA train + checkpoint + eval",
        ],
        "#d9f4e9",
    ),
    (
        640,
        "03  CHECK",
        "Executable evidence",
        [
            "Capabilities and dependencies",
            "Topology and optimizer state",
            "Independent mechanism probes",
        ],
        "#fff0da",
    ),
    (
        940,
        "04  REPORT",
        "Reviewable results",
        [
            "Source and dataset hashes",
            "Per-view quality + resources",
            "Signed, absolute, worst gaps",
        ],
        "#e2edff",
    ),
]
for x, label, heading, lines, color in cards:
    body += (
        f'<rect x="{x}" y="128" width="260" height="225" rx="16" fill="white" stroke="#d7e0e8"/>'
    )
    body += f'<rect x="{x + 18}" y="148" width="224" height="29" rx="8" fill="{color}"/>'
    body += text(x + 30, 168, label, 11, "#34465c", 700)
    body += text(x + 18, 215, heading, 19, "#17243a", 700)
    for i, line in enumerate(lines):
        body += text(x + 18, 253 + i * 28, line, 13, "#4e6077")
    if x < 940:
        body += f'<path d="M{x + 270} 240 h20 l-5 -5 m5 5 l-5 5" fill="none" stroke="#6d819b" stroke-width="2"/>'
body += '<rect x="40" y="386" width="560" height="118" rx="16" fill="#17243a"/>'
body += text(62, 416, "NERFSTUDIO COLLECTION", 11, "#8ce4ca", 700)
body += text(62, 447, "33 installable research configurations", 19, "#f4f6fc", 600)
body += text(
    62, 477, "pip install -e packages/nerfstudio-methods → ns-train ss-absgs", 13, "#c1ccdc"
)
body += '<rect x="640" y="386" width="560" height="118" rx="16" fill="#eff3f7" stroke="#d7e0e8"/>'
body += text(662, 418, "STATUS HAS A PRECISE MEANING", 11, "#596b80", 700)
body += text(
    662, 449, "Import → short training → mechanism tests → reproduction", 17, "#17243a", 600
)
body += text(
    662,
    478,
    "Each stage needs its own evidence. Passing one does not establish the next.",
    13,
    "#596b80",
)
save("architecture.svg", 1240, 540, body, "#f7f9fc")
print("Wrote docs/assets/hero.svg and architecture.svg")
