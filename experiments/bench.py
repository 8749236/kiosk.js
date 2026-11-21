"""Compare invisible-watermark algorithms on illustration-like images.

Every detector is given the original image (the artist always has it), so the
leaked copy is first registered back onto the original's pixel grid.
Run:  python experiments/bench.py
"""
import io, os, sys, glob, json, warnings, zlib
import numpy as np, cv2
from PIL import Image

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
NBITS = 32
rng = np.random.default_rng(7)

# ---------------------------------------------------------------- images
def synth_cel(h=1024, w=1024):
    """Flat colour regions + black line art + one soft gradient: cel-shaded stand-in."""
    r = np.random.default_rng(1)
    img = np.full((h, w, 3), 255, np.uint8)
    sky = np.linspace(0, 1, h)[:, None]
    img[:, :] = (np.array([255, 200, 150]) * (1 - sky[..., None]) + np.array([255, 240, 230]) * sky[..., None]).astype(np.uint8)
    for _ in range(25):
        c = tuple(int(x) for x in r.integers(30, 255, 3))
        pts = r.integers(0, w, (r.integers(3, 7), 2)).astype(np.int32)
        cv2.fillPoly(img, [pts], c, lineType=cv2.LINE_AA)
        cv2.polylines(img, [pts], True, (20, 20, 20), 3, lineType=cv2.LINE_AA)
    for _ in range(12):
        c = tuple(int(x) for x in r.integers(30, 255, 3))
        p = tuple(int(x) for x in r.integers(100, w - 100, 2)); rad = int(r.integers(30, 140))
        cv2.circle(img, p, rad, c, -1, lineType=cv2.LINE_AA); cv2.circle(img, p, rad, (20, 20, 20), 3, lineType=cv2.LINE_AA)
    return img

def synth_lineart(h=1024, w=1024):
    r = np.random.default_rng(2)
    img = np.full((h, w, 3), 255, np.uint8)
    for _ in range(60):
        pts = np.cumsum(r.normal(0, 60, (12, 2)), 0) + r.integers(100, w - 100, 2)
        cv2.polylines(img, [pts.astype(np.int32)], False, (0, 0, 0), int(r.integers(1, 4)), lineType=cv2.LINE_AA)
    return img

def load_images():
    ims = {}
    for name, bgr in [("white", (255, 255, 255)), ("black", (0, 0, 0)), ("red", (0, 0, 255)),
                      ("green", (0, 255, 0)), ("blue", (255, 0, 0))]:
        ims["solid_" + name] = np.full((1024, 1024, 3), bgr, np.uint8)
    ims["synth_cel"] = synth_cel(); ims["synth_lineart"] = synth_lineart()
    for p in sorted(glob.glob(os.path.join(HERE, "real", "*"))):
        im = Image.open(p)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA"); bg = Image.new("RGBA", im.size, (255, 255, 255, 255)); im = Image.alpha_composite(bg, im)
        a = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
        s = 1280 / max(a.shape[:2])
        if s < 1: a = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        a = a[: a.shape[0] // 16 * 16, : a.shape[1] // 16 * 16]
        ims["real_" + os.path.splitext(os.path.basename(p))[0][:14].lower()] = np.ascontiguousarray(a)
    return ims

# ---------------------------------------------------------------- attacks
def jpeg(img, q):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q]); return cv2.imdecode(buf, 1)
def resize(img, s):
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
def crop(img, f=0.12):
    h, w = img.shape[:2]; return img[int(h * f):, int(w * f): int(w * (1 - f / 2))]
def tone(img):
    x = img.astype(np.float32) / 255; x = np.clip(1.08 * x ** 0.92 + 0.02, 0, 1); return (x * 255).round().astype(np.uint8)
def noise(img, s=3):
    return np.clip(img + np.random.default_rng(3).normal(0, s, img.shape), 0, 255).round().astype(np.uint8)

def webp(img, q):
    ok, buf = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, q]); return cv2.imdecode(buf, 1)
def center_crop(img, f=0.5):
    h, w = img.shape[:2]; y, x = int(h * (1 - f) / 2), int(w * (1 - f) / 2); return img[y: y + int(h * f), x: x + int(w * f)]

ATTACKS = {
    "none": lambda i: i,
    "jpeg90": lambda i: jpeg(i, 90),
    "jpeg75": lambda i: jpeg(i, 75),
    "jpeg50": lambda i: jpeg(i, 50),
    "webp80": lambda i: webp(i, 80),
    "resize50": lambda i: resize(i, 0.5),
    "resize30+jpeg": lambda i: jpeg(resize(i, 0.3), 85),
    "screenshot": lambda i: jpeg(tone(resize(i, 0.7)), 85),   # downscaled, tone-shifted, recompressed
    "crop+jpeg": lambda i: jpeg(crop(i), 85),
    "crop+resize": lambda i: jpeg(resize(crop(i), 0.6), 85),
    "charcrop": lambda i: jpeg(center_crop(i), 85),          # keep only the middle half (the character)
    "charcrop+resize+webp": lambda i: webp(resize(center_crop(i), 0.6), 80),
    "noise": lambda i: noise(i),
}

# ---------------------------------------------------------------- registration
def register(leak, orig):
    """Map a leaked copy back onto the original's pixel grid (scale + translation)."""
    H, W = orig.shape[:2]; h, w = leak.shape[:2]
    M = None
    try:
        sift = cv2.SIFT_create(4000)
        k1, d1 = sift.detectAndCompute(cv2.cvtColor(leak, cv2.COLOR_BGR2GRAY), None)
        k2, d2 = sift.detectAndCompute(cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY), None)
        if d1 is not None and d2 is not None and len(k1) > 20 and len(k2) > 20:
            m = cv2.BFMatcher().knnMatch(d1, d2, k=2)
            good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.7 * b.distance]
            if len(good) >= 12:
                src = np.float32([k1[g.queryIdx].pt for g in good]); dst = np.float32([k2[g.trainIdx].pt for g in good])
                A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
                if A is not None and inl.sum() >= 10:
                    s = (A[0, 0] + A[1, 1]) / 2   # drop rotation: keep uniform scale + shift
                    M = np.float32([[s, 0, A[0, 2]], [0, s, A[1, 2]]])
    except cv2.error:
        pass
    if M is None:   # featureless image: assume a plain resize
        M = np.float32([[W / w, 0, 0], [0, H / h, 0]])
        if abs(W / w - 1) < 1e-6 and abs(H / h - 1) < 1e-6: return leak.copy(), np.ones((H, W), bool)
    interp = cv2.INTER_CUBIC
    warped = cv2.warpAffine(leak, M, (W, H), flags=interp, borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (W, H), flags=cv2.INTER_NEAREST) > 0
    out = orig.copy(); out[mask] = warped[mask]      # uncovered area: fill with original (carries no signal)
    return out, mask

# ---------------------------------------------------------------- algorithms
class BlindWM:
    name = "blind_watermark"
    def embed(self, img, bits):
        from blind_watermark import WaterMark
        bw = WaterMark(password_img=1, password_wm=1); bw.read_img(img=img.copy()); bw.read_wm(bits.astype(bool), mode="bit")
        return np.clip(bw.embed(), 0, 255).round().astype(np.uint8)
    def detect(self, reg, orig, mask, leak=None):
        from blind_watermark import WaterMark
        bw = WaterMark(password_img=1, password_wm=1)
        return (np.asarray(bw.extract(embed_img=reg, wm_shape=NBITS, mode="bit")) > 0.5).astype(int)

class BlindWMRaw(BlindWM):
    """blind_watermark as shipped: no original available, leak is only stretched back to size."""
    name = "blind_watermark (no registration)"
    def detect(self, reg, orig, mask, leak=None):
        return super().detect(cv2.resize(leak, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_CUBIC), orig, mask)

class InvWM:
    def __init__(self, method): self.method = method; self.name = "invisible-wm/" + method
    def embed(self, img, bits):
        from imwatermark import WatermarkEncoder
        e = WatermarkEncoder(); e.set_watermark("bits", [int(b) for b in bits]); return e.encode(img.copy(), self.method)
    def detect(self, reg, orig, mask, leak=None):
        from imwatermark import WatermarkDecoder
        return np.asarray(WatermarkDecoder("bits", NBITS).decode(reg, self.method)).astype(int)

class Trust:
    name = "trustmark-Q"
    def __init__(self):
        from trustmark import TrustMark
        self.tm = TrustMark(verbose=False, model_type="Q", encoding_type=TrustMark.Encoding.BCH_5, loadRemover=False)
        self.n = self.tm.schemaCapacity()
    def embed(self, img, bits):
        s = "".join(map(str, bits)) + "0" * (self.n - NBITS)
        out = self.tm.encode(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), s, MODE="binary")
        return cv2.cvtColor(np.array(out.convert("RGB")), cv2.COLOR_RGB2BGR)
    def detect(self, reg, orig, mask, leak=None):
        s, present, _ = self.tm.decode(Image.fromarray(cv2.cvtColor(reg, cv2.COLOR_BGR2RGB)), MODE="binary")
        if not present or not s: return np.full(NBITS, -1)       # ECC failed: nothing readable
        return np.array([int(c) for c in s[:NBITS]])

class NonBlind:
    """Ours: block spread-spectrum on luma, read by subtracting the original.

    The image is split into a GxG grid (relative to image size, so it survives
    rescaling). Each cell is assigned to one payload bit and shifted by +-amp
    grey levels. Detection = (leak - original), correlated with the known signs.
    """
    def __init__(self, amp, grid=48, adaptive=0.0):
        self.amp, self.grid, self.adaptive = amp, grid, adaptive
        self.name = f"nonblind a={amp}" + (f"+tex{adaptive:g}" if adaptive else "")
    def _layout(self, shape, key=1234):
        H, W = shape[:2]; G = self.grid; r = np.random.default_rng(key)
        owner = r.permutation(G * G) % NBITS; sign = r.choice([-1.0, 1.0], G * G)
        ys = np.minimum(np.arange(H) * G // H, G - 1); xs = np.minimum(np.arange(W) * G // W, G - 1)
        cell = ys[:, None] * G + xs[None, :]
        return owner[cell], sign[cell]
    def embed(self, img, bits):
        owner, sign = self._layout(img.shape)
        pat = sign * (2.0 * bits[owner] - 1)
        amp = np.full(img.shape[:2], float(self.amp))
        if self.adaptive:   # hide more where there is texture / line work
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            sd = np.sqrt(np.maximum(cv2.blur(g * g, (9, 9)) - cv2.blur(g, (9, 9)) ** 2, 0))
            amp += np.minimum(self.adaptive * sd / 8, 4 * self.adaptive)
        d = np.round(pat * amp)
        return np.clip(img.astype(np.float32) + d[..., None], 0, 255).astype(np.uint8)
    def detect(self, reg, orig, mask, leak=None):
        owner, sign = self._layout(orig.shape)
        d = np.zeros(orig.shape[:2], np.float32)
        for c in range(3):   # undo tone/colour shifts: learn leak = f(original) per channel, then subtract
            o = orig[..., c].ravel(); l = reg[..., c].ravel().astype(np.float64)
            n = np.bincount(o[mask.ravel()], minlength=256); t = np.bincount(o[mask.ravel()], l[mask.ravel()], 256)
            lut = np.where(n > 0, t / np.maximum(n, 1), np.arange(256))
            d += (reg[..., c] - lut[orig[..., c]]).astype(np.float32) / 3
        d = np.clip(d, -12, 12) * mask                                           # ignore gross mismatches
        score = np.bincount(owner.ravel(), (d * sign).ravel(), NBITS)
        return (score > 0).astype(int)

class NonBlindSmooth(NonBlind):
    """Same payload layout, but the +-1 cell grid is interpolated into a smooth field
    and rounded with random dither, so there are no block edges or contour lines."""
    def __init__(self, amp, grid=48, adaptive=0.0):
        super().__init__(amp, grid, adaptive); self.name = f"smooth a={amp:g}" + (f"+tex{adaptive:g}" if adaptive else "")
    def _field(self, shape, bits, key=1234):
        H, W = shape[:2]; G = self.grid; r = np.random.default_rng(key)
        owner = r.permutation(G * G) % NBITS; sign = r.choice([-1.0, 1.0], G * G)
        cells = (sign * (2.0 * bits[owner] - 1)).reshape(G, G).astype(np.float32)
        return np.clip(cv2.resize(cells, (W, H), interpolation=cv2.INTER_CUBIC), -1, 1)
    def embed(self, img, bits):
        f = self._field(img.shape, bits)
        amp = np.full(img.shape[:2], float(self.amp), np.float32)
        if self.adaptive:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            sd = np.sqrt(np.maximum(cv2.blur(g * g, (9, 9)) - cv2.blur(g, (9, 9)) ** 2, 0))
            amp += np.minimum(self.adaptive * sd / 8, 4 * self.adaptive)
        v = f * amp
        d = np.floor(v + np.random.default_rng(99).random(v.shape))      # dithered rounding
        return np.clip(img.astype(np.float32) + d[..., None], 0, 255).astype(np.uint8)

# ---------------------------------------------------------------- run
def psnr(a, b):
    m = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2); return 99.0 if m == 0 else 10 * np.log10(255 ** 2 / m)

def main():
    ims = load_images()
    algos = [BlindWM(), BlindWMRaw(), InvWM("dwtDct"), InvWM("dwtDctSvd"), Trust(),
             NonBlind(1), NonBlind(2), NonBlind(3), NonBlind(1, adaptive=1.0),
             NonBlindSmooth(1), NonBlindSmooth(2), NonBlindSmooth(3), NonBlindSmooth(1, adaptive=1.0)]
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--no-solid" in sys.argv: ims = {k: v for k, v in ims.items() if not k.startswith("solid") or k == "solid_white"}
    rp = os.path.join(OUT, 'results.json')
    res = json.load(open(rp)) if only and os.path.exists(rp) else {}
    for k in [k for k in res if any(k.startswith(o) for o in only)]: del res[k]
    for al in algos:
        if only and not any(al.name.startswith(o) for o in only): continue
        for iname, img in ims.items():
            payloads = [np.random.default_rng(zlib.crc32(iname.encode()) + k).integers(0, 2, NBITS) for k in range(2)]
            for k, bits in enumerate(payloads):
                try:
                    wm = al.embed(img, bits)
                except Exception as e:
                    res.setdefault(al.name, {}).setdefault(iname, {})["error"] = repr(e)[:80]; break
                r = res.setdefault(al.name, {}).setdefault(iname, {"psnr": [], "maxdiff": [], "acc": {a: [] for a in ATTACKS}})
                r["psnr"].append(float(psnr(img, wm))); r["maxdiff"].append(int(np.abs(img.astype(int) - wm.astype(int)).max()))
                if k == 0:
                    cv2.imwrite(os.path.join(OUT, f"{iname}__{al.name.replace('/', '_').replace(' ', '_')}.png"), wm)
                for an, fn in ATTACKS.items():
                    try:
                        reg, mask = register(fn(wm), img)
                        got = al.detect(reg, img, mask, fn(wm)); acc = float((got == bits).mean())
                    except Exception as e:
                        acc = float("nan")
                    r["acc"][an].append(acc)
            print(al.name, iname, "done", flush=True)
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
