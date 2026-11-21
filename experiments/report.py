"""Turn out/results.json into a markdown summary (experiments/RESULTS.md)."""
import json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(os.path.join(HERE, "out", "results.json")))
groups = {"solid": "纯色图", "synth": "合成插画", "real": "真实插画"}
lines = []
def m(v):
    v = [x for x in v if x == x]; return float(np.mean(v)) if v else float("nan")
for g, title in groups.items():
    attacks = None; rows = []
    for al, per in res.items():
        sel = {k: v for k, v in per.items() if k.startswith(g) and "acc" in v}
        if not sel: continue
        attacks = attacks or list(next(iter(sel.values()))["acc"])
        ps = m([m(v["psnr"]) for v in sel.values()]); md = max(max(v["maxdiff"]) for v in sel.values())
        accs = [m([m(v["acc"][a]) for v in sel.values()]) for a in attacks]
        rows.append(f"| {al} | {ps:.1f} | {md} | " + " | ".join(f"{100*a:.0f}" for a in accs) + " |")
    lines += [f"### {title}", "", "| 算法 | PSNR | 最大像素差 | " + " | ".join(attacks) + " |", "|---|---|---|" + "---|" * len(attacks)] + rows + [""]
open(os.path.join(HERE, "out", "summary.md"), "w").write("\n".join(lines)); print("\n".join(lines))
