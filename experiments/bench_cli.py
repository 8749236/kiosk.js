"""End-to-end check of the Rust CLI: embed, damage the copy the way re-uploaders do, detect.
Run:  cargo build --release && python experiments/bench_cli.py
"""
import json, os, subprocess, sys, time, tempfile
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench

BIN = os.path.join(os.path.dirname(bench.HERE), "target", "release", "tracemark")
ATTACKS = dict(bench.ATTACKS)
ATTACKS["mirror+jpeg"] = lambda i: bench.jpeg(cv2.flip(i, 1), 85)
ATTACKS["mirror+charcrop+resize"] = lambda i: bench.jpeg(bench.resize(bench.center_crop(cv2.flip(i, 1)), 0.6), 85)
IDS = [7, 3141592653]
DECOYS = ",".join(str(1000 + i) for i in range(200))

def main():
    ims = {k: v for k, v in bench.load_images().items() if not k.startswith("solid") or k == "solid_white"}
    tmp = tempfile.mkdtemp()
    rows, t_embed, t_detect = {}, [], []
    for name, img in ims.items():
        orig = os.path.join(tmp, "orig.png"); cv2.imwrite(orig, img)
        for rid in IDS:
            marked = os.path.join(tmp, "marked.png")
            t = time.time(); subprocess.run([BIN, "embed", orig, marked, "--id", str(rid)], check=True, capture_output=True); t_embed.append(time.time() - t)
            wm = cv2.imread(marked)
            for an, fn in ATTACKS.items():
                leak = os.path.join(tmp, "leak.png"); cv2.imwrite(leak, fn(wm))
                t = time.time()
                blind = json.loads(subprocess.run([BIN, "detect", orig, leak, "--json"], capture_output=True, text=True).stdout)
                t_detect.append(time.time() - t)
                listed = json.loads(subprocess.run([BIN, "detect", orig, leak, "--json", "--id", f"{DECOYS},{rid}"], capture_output=True, text=True).stdout)
                r = rows.setdefault(name, {}).setdefault(an, {"blind": 0, "listed": 0, "wrong": 0, "strength": []})
                for mode, d in (("blind", blind), ("listed", listed)):
                    if d["verdict"] == "match":
                        if d["id"] == rid: r[mode] += 1
                        else: r["wrong"] += 1
                r["strength"].append(listed["strength"])
        # an unmarked copy must never name anyone
        leak = os.path.join(tmp, "leak.png"); cv2.imwrite(leak, bench.jpeg(bench.resize(img, 0.7), 85))
        d = json.loads(subprocess.run([BIN, "detect", orig, leak, "--json", "--id", DECOYS], capture_output=True, text=True).stdout)
        rows[name]["_clean"] = d["verdict"]
    attacks = list(ATTACKS)
    print("每格: 直接读出 / 配合已发放名单 (各 %d 次), 括号内为平均强度\n" % len(IDS))
    print("| 图片 | " + " | ".join(attacks) + " | 未加水印的图 |"); print("|---|" + "---|" * (len(attacks) + 1))
    wrong = 0
    for name, r in rows.items():
        cells = [f"{r[a]['blind']}/{r[a]['listed']} ({sum(r[a]['strength'])/len(r[a]['strength']):.0f})" for a in attacks]
        wrong += sum(r[a]["wrong"] for a in attacks)
        print(f"| {name} | " + " | ".join(cells) + f" | {r['_clean']} |")
    print(f"\n认错人的次数: {wrong}")
    print(f"耗时: 嵌入平均 {sum(t_embed)/len(t_embed):.2f}s, 识别平均 {sum(t_detect)/len(t_detect):.2f}s")

if __name__ == "__main__":
    main()
