"""Push the Rust CLI until it breaks. Run: cargo build --release && python experiments/stress_cli.py"""
import sys, os, json, subprocess, tempfile, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench
BIN = os.path.join(os.path.dirname(bench.HERE), "target", "release", "tracemark")
def rotate(i, deg):
    m = cv2.getRotationMatrix2D((i.shape[1] / 2, i.shape[0] / 2), deg, 1)
    return cv2.warpAffine(i, m, (i.shape[1], i.shape[0]), borderValue=(255, 255, 255))
def saturate(i):
    hsv = cv2.cvtColor(i, cv2.COLOR_BGR2HSV).astype(np.float32) * [1, 1.3, 1]
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
STRESS = {
    "缩到 20%": lambda i: bench.jpeg(bench.resize(i, 0.2), 85),
    "缩到 12%": lambda i: bench.jpeg(bench.resize(i, 0.12), 85),
    "JPEG 质量 30": lambda i: bench.jpeg(i, 30),
    "JPEG 质量 15": lambda i: bench.jpeg(i, 15),
    "只留 1/4 面积": lambda i: bench.jpeg(bench.center_crop(i, 0.5), 85),
    "只留 1/9 面积": lambda i: bench.jpeg(bench.center_crop(i, 0.33), 85),
    "只留 1/16 面积": lambda i: bench.jpeg(bench.center_crop(i, 0.25), 85),
    "只留 1/9 再缩一半": lambda i: bench.jpeg(bench.resize(bench.center_crop(i, 0.33), 0.5), 80),
    "横向拉宽 10%": lambda i: bench.jpeg(cv2.resize(i, None, fx=1.1, fy=1.0), 90),
    "模糊": lambda i: bench.jpeg(cv2.GaussianBlur(i, (0, 0), 1.5), 85),
    "加饱和度": lambda i: bench.jpeg(saturate(i), 85),
    "贴一块色块遮挡": lambda i: bench.jpeg(cv2.rectangle(i.copy(), (50, 50), (i.shape[1] // 3, i.shape[0] // 4), (0, 0, 255), -1), 85),
    "旋转 0.5 度": lambda i: bench.jpeg(rotate(i, 0.5), 90),
    "旋转 2 度": lambda i: bench.jpeg(rotate(i, 2), 90),
}
ims = {k: v for k, v in bench.load_images().items() if k.startswith("real")}
tmp = tempfile.mkdtemp(); res = {a: [0, 0, []] for a in STRESS}
ids = ",".join(map(str, range(4000, 4243)))
for name, img in ims.items():
    o, m, l = (os.path.join(tmp, f) for f in ("o.png", "m.png", "l.png")); cv2.imwrite(o, img)
    subprocess.run([BIN, "embed", o, m, "--id", "4242"], check=True, capture_output=True); wm = cv2.imread(m)
    for a, fn in STRESS.items():
        cv2.imwrite(l, fn(wm))
        b = json.loads(subprocess.run([BIN, "detect", o, l, "--json"], capture_output=True, text=True).stdout)
        c = json.loads(subprocess.run([BIN, "detect", o, l, "--json", "--id", ids], capture_output=True, text=True).stdout)
        res[a][0] += b["verdict"] == "match" and b["id"] == 4242
        res[a][1] += c["verdict"] == "match" and c["id"] == 4242
        res[a][2].append(c["strength"])
print(f"| 破坏方式 | 直接读出 | 配合名单 | 强度 |\n|---|---|---|---|")
for a, (b, c, s) in res.items(): print(f"| {a} | {b}/{len(ims)} | {c}/{len(ims)} | {min(s):.0f}–{max(s):.0f} |")
