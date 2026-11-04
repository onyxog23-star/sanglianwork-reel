# Sanglianwork ASCII reel — run:  python make_reel.py   (or: python make_reel.py test  for 2 preview images)
import numpy as np, re, subprocess, sys
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

import os
HERE = os.path.dirname(os.path.abspath(__file__))

# ===================== EDIT THESE =====================
LETTER   = "S"                       # the shape that gets drawn (try "SW" or "L")
SITE     = "sanglianwork.de"         # top-left label
HANDLE   = "@sanglianwork · 2026"    # bottom-left label
SCRIPT   = "py sanglianwork.py"      # label above the code
OUTPUT   = "sanglianwork_reel.mp4"
DUR      = 20.0                      # seconds
# =====================================================

W, H, FPS = 1080, 1920, 30
TAU = 2 * np.pi
MONO = os.path.join(HERE, "fonts", "DejaVuSansMono.ttf")
BOLD = os.path.join(HERE, "fonts", "Poppins-Bold.ttf")

BG = np.array([6, 10, 8], float)
MINT = np.array([226, 240, 231], float)
LIME = np.array([198, 244, 50], float)
DIM = np.array([120, 146, 132], float)

# ---------- glyph atlas ----------
CW, CH = 8, 13
COLS, ROWS = W // CW, 81
RY0 = 96                                 # render area top
RAMP = " .:-=+*#%@"
EXTRA = ["+", "*", "x", "·", "|", "-", "o"]
CHARS = list(RAMP) + EXTRA
CI = {c: i for i, c in enumerate(CHARS)}
gfont = ImageFont.truetype(MONO, 12)
atlas = np.zeros((len(CHARS), CH, CW))
for i, c in enumerate(CHARS):
    im = Image.new("L", (CW, CH), 0)
    ImageDraw.Draw(im).text((CW / 2, CH / 2), c, 255, font=gfont, anchor="mm")
    atlas[i] = np.asarray(im) / 255.0

# ---------- shape ----------
def glyph_mask(size=300):
    f = ImageFont.truetype(BOLD, size)
    im = Image.new("L", (size * 2, size * 2), 0)
    im = Image.new("L", (size * 2 * max(1, len(LETTER)), size * 2), 0)
    ImageDraw.Draw(im).text((im.width // 2, size), LETTER, 255, font=f, anchor="mm")
    m = np.asarray(im) > 127
    ys, xs = np.nonzero(m)
    return m[ys.min() - 3:ys.max() + 4, xs.min() - 3:xs.max() + 4]

mask = glyph_mask()
mh, mw = mask.shape
scale = min(2.0 / mh, 2.5 / mw)
DEPTH = 0.2
ys, xs = np.nonzero(mask)
p2 = np.stack([(xs - mw / 2) * scale, -(ys - mh / 2) * scale], 1)
# subsample faces so density ~ 2 pts per cell
sub = 2
sel = (ys % sub == 0) & (xs % sub == 0)
fp = p2[sel]
front = np.c_[fp, np.full(len(fp), DEPTH)]
back = np.c_[fp, np.full(len(fp), -DEPTH)]
pb = ndimage.gaussian_filter(mask.astype(float), 7)
pgy, pgx = np.gradient(pb)
fy, fx = ys[sel], xs[sel]
k_b = 9.0
nf = np.stack([-pgx[fy, fx] * k_b, pgy[fy, fx] * k_b, np.ones(len(fy))], 1)
nf /= np.linalg.norm(nf, axis=1, keepdims=True)
nb = nf * np.array([1, 1, -1.0])
edge = mask & ~ndimage.binary_erosion(mask, iterations=2)
blur = ndimage.gaussian_filter(mask.astype(float), 3)
gy, gx = np.gradient(blur)
ey, ex = np.nonzero(edge)
k = (ey + ex) % 2 == 0
ey, ex = ey[k], ex[k]
en = np.stack([-gx[ey, ex], gy[ey, ex], np.zeros(len(ey))], 1)
en /= np.linalg.norm(en, axis=1, keepdims=True) + 1e-9
zs = np.linspace(-DEPTH, DEPTH, 12)
ep = np.stack([(ex - mw / 2) * scale, -(ey - mh / 2) * scale], 1)
wall = np.concatenate([np.c_[ep, np.full(len(ep), z)] for z in zs])
wn = np.tile(en, (len(zs), 1))
# bevel: blend normals near the edge on faces for a softer metal look
SHAPE = np.concatenate([front, back, wall])
NORM = np.concatenate([nf, nb, wn])

# stroke parameter (pen path through the S, bottom-left tail → top-right tail)
def stroke_param(p):
    x, y = p[:, 0], p[:, 1]
    r = 0.47
    low = y < 0
    a = np.arctan2(y + r, x)                 # lower bowl, ccw from bottom-left
    a = np.where(a > 2.4, a - TAU, a)        # tail wraps
    sl = (a - (-3.0)) / (np.pi / 2 + 3.0)    # -3.0 .. +pi/2
    b = (-np.pi / 2 - np.arctan2(y - r, x)) % TAU   # upper bowl, cw from bottom
    b = np.where(b > 5.6, b - TAU, b)
    su = b / 4.7
    s = np.where(low, sl * 0.5, 0.5 + su * 0.5)
    return np.clip(s, 0, 1)

_sp = stroke_param(SHAPE[:, :2]) if LETTER == "S" else np.clip(0.75 * (SHAPE[:, 0] - SHAPE[:, 0].min()) / np.ptp(SHAPE[:, 0]) + 0.25 * (SHAPE[:, 1] - SHAPE[:, 1].min()) / np.ptp(SHAPE[:, 1]), 0, 1)
STROKE = _sp + 0.004 * np.random.default_rng(3).standard_normal(len(SHAPE))
RNG = np.random.default_rng(7)
PHASE = RNG.random(len(SHAPE)) * TAU

# top-right terminal (for the star burst)
tt = np.argmax(STROKE * (SHAPE[:, 2] > DEPTH - 1e-6))
TIP = SHAPE[tt].copy()

# ---------- orbits ----------
def rot_x(p, a):
    c, s = np.cos(a), np.sin(a)
    return np.stack([p[:, 0], c * p[:, 1] - s * p[:, 2], s * p[:, 1] + c * p[:, 2]], 1)
def rot_y(p, a):
    c, s = np.cos(a), np.sin(a)
    return np.stack([c * p[:, 0] + s * p[:, 2], p[:, 1], -s * p[:, 0] + c * p[:, 2]], 1)
def rot_z(p, a):
    c, s = np.cos(a), np.sin(a)
    return np.stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1], p[:, 2]], 1)

NO = 520
OA = np.linspace(0, TAU, NO, endpoint=False)
def orbit_pts(tilt, t, R=1.32):
    a = OA + t * 0.35 * np.sign(tilt)
    ring = np.c_[R * np.cos(a), 0 * a, R * 0.62 * np.sin(a)]
    return rot_z(rot_x(ring, 0.22), tilt)

# ---------- dust ----------
ND = 260
DX = RNG.integers(0, COLS, ND); DY = RNG.integers(0, ROWS - 12, ND)
DPH = RNG.random(ND) * TAU; DSP = 0.5 + RNG.random(ND) * 1.5

# ---------- helpers ----------
CAM, F = 7.0, 2500.0
CX, CY = W / 2, 500.0              # S centre inside render area (px)
def project(p):
    w = CAM - p[:, 2]
    return CX + F * p[:, 0] / w, CY - F * p[:, 1] / w, w

def smooth(x):
    x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)

def spin(t):
    u = max(t - 5.4, 0)
    return 0.55 * u if u > 1.2 else 0.55 * (u * u / 2.4)  # ease in, then constant

key = np.array([-0.4, 0.55, 0.73]); key /= np.linalg.norm(key)
DRAW = (0.35, 5.0)

# ---------- code panel ----------
CODE = open(os.path.join(HERE, "code_shown.py"), encoding="utf-8").read().rstrip("\n").split("\n")
FULL = "\n".join(CODE)
cfont = ImageFont.truetype(MONO, 23)
sfont = ImageFont.truetype(MONO, 17)
CODE_Y0, LH = 1192, 33
VIS = (H - CODE_Y0 - 10) // LH
CCW = cfont.getlength("M")
KW = r"\b(def|return|for|in|if|else|import|from|as|lambda|None|True|False|and|or|not|yield)\b"
TOK = re.compile(r"(?P<c>#.*)|(?P<s>\"\"\".*?\"\"\"|\"[^\"]*\"|'[^']*')|(?P<k>" + KW + r")|(?P<n>\b\d+\.?\d*\b)|(?P<f>\b[A-Za-z_]\w*(?=\())|(?P<m>\bnp\b)")
COLS_SYN = {"c": (88, 116, 100), "s": (242, 193, 78), "k": (198, 244, 50), "n": (247, 140, 108),
            "f": (127, 216, 190), "m": (170, 190, 180), None: (205, 218, 210)}
def tokens(line):
    out, i = [], 0
    for m in TOK.finditer(line):
        if m.start() > i: out.append((line[i:m.start()], None))
        out.append((m.group(0), m.lastgroup)); i = m.end()
    if i < len(line): out.append((line[i:], None))
    return out
TYPE_END = 18.6

def draw_code(d, t):
    n = int(len(FULL) * smooth(min(max(t - 0.1, 0) / TYPE_END, 1)) ** 1.0) if t < TYPE_END else len(FULL)
    n = int(len(FULL) * min(max(t - 0.1, 0) / TYPE_END, 1))
    typed = FULL[:n].split("\n")
    cur_line = len(typed) - 1
    top = max(0, cur_line - (VIS - 3))
    for r in range(VIS):
        li = top + r
        if li > cur_line: break
        y = CODE_Y0 + r * LH
        txt = typed[li]
        if li == cur_line:
            d.rectangle([0, y - 3, W, y + LH - 6], fill=(18, 30, 16))
        d.text((14, y), f"{li + 1:>3}", font=cfont, fill=(56, 74, 64))
        x = 80
        for s, k in tokens(txt):
            d.text((x, y), s, font=cfont, fill=COLS_SYN[k]); x += cfont.getlength(s)
        if li == cur_line and int(t * 2.4) % 2 == 0 or (li == cur_line and n < len(FULL)):
            d.rectangle([x + 1, y + 1, x + CCW - 2, y + LH - 9], fill=(198, 244, 50))

# ---------- frame ----------
def frame(t):
    G = np.zeros((ROWS, COLS), int)
    C = np.zeros((ROWS, COLS, 3))
    # dust
    tw = 0.5 + 0.5 * np.sin(t * DSP + DPH)
    dy = (DY - (t * DSP * 0.6).astype(int)) % (ROWS - 12)
    on = tw > 0.55
    G[dy[on], DX[on]] = CI["·"]
    C[dy[on], DX[on]] = DIM * 0.35 * tw[on, None]

    ang = spin(t)
    tilt = 0.10 * np.sin(t * 0.6) * smooth((t - 5) / 2)
    P = rot_x(rot_y(SHAPE, ang), tilt)
    N = rot_x(rot_y(NORM, ang), tilt)
    head = (t - DRAW[0]) / (DRAW[1] - DRAW[0]) * 1.06
    shown = STROKE < head
    spark = shown & (head - STROKE < 0.03) & (t < DRAW[1] + 0.2)

    rim = np.array([np.cos(t * 0.9), 0.35, np.sin(t * 0.9)]); rim /= np.linalg.norm(rim)
    v = np.array([0, 0, 1.0])
    diff = np.clip(N @ key, 0, 1)
    hv = key + v; hv /= np.linalg.norm(hv)
    spec = np.clip(N @ hv, 0, 1) ** 24
    rimv = np.clip(N @ rim, 0, 1) ** 3
    b = 0.12 + 0.70 * diff + 0.55 * rimv + 0.6 * spec
    # landing flash
    age = np.clip(head - STROKE, 0, 1)
    b = b + 0.5 * np.exp(-age * 40) * shown

    # orbits
    oreveal = smooth((t - 5.2) / 1.8)
    opts, ochar, ocol = [], [], []
    for tilt_o in (0.62, -0.62):
        op = rot_x(orbit_pts(tilt_o, t), tilt)
        frac = (np.arange(NO) / NO)
        vis = frac < oreveal
        # traveling comet
        comet = np.exp(-((((OA - t * 1.3 * np.sign(tilt_o)) % TAU) - np.pi) ** 2) / 0.04)
        dash = (np.arange(NO) % 3 != 0)
        vis &= dash | (comet > 0.2)
        opts.append(op[vis]);
        cc = np.where(comet[vis] > 0.3, CI["o"], CI["·"])
        ochar.append(cc)
        ocol.append(DIM[None] * (0.75 + 1.2 * comet[vis, None]).clip(0, 1.6))
    if oreveal > 0:
        OP = np.concatenate(opts); OCH = np.concatenate(ochar); OCOL = np.concatenate(ocol)
    else:
        OP = np.zeros((0, 3)); OCH = np.zeros(0, int); OCOL = np.zeros((0, 3))

    # S chars + colors
    Ps = P[shown]; bs = b[shown]
    ri = np.clip((bs * (len(RAMP) - 1)), 1, len(RAMP) - 1).astype(int)
    sch = ri
    scol = MINT[None] * np.clip(0.38 + 0.8 * bs, 0, 1.08)[:, None]
    sp = spark[shown]
    sch = np.where(sp, CI["*"], sch); scol[sp] = LIME
    glint = (spec[shown] > 0.45) & (np.sin(t * 7 + PHASE[shown]) > 0.6)
    sch = np.where(glint & ~sp, CI["+"], sch); scol[glint & ~sp] = np.array([236, 255, 170])
    n_glint = int(glint.sum())

    AP = np.concatenate([OP, Ps]); ACH = np.concatenate([OCH, sch]); ACOL = np.concatenate([OCOL, scol])
    isS = np.r_[np.zeros(len(OP), bool), np.ones(len(Ps), bool)]
    if len(AP):
        x, y, w = project(AP)
        cx = (x / CW).astype(int); cy = (y / CH).astype(int)
        ok = (cx >= 0) & (cx < COLS) & (cy >= 0) & (cy < ROWS)
        # depth: S slightly preferred
        depth = w - isS * 0.05
        idx = np.nonzero(ok)[0]
        idx = idx[np.argsort(-depth[idx])]
        G[cy[idx], cx[idx]] = ACH[idx]
        C[cy[idx], cx[idx]] = ACOL[idx]
        filled = np.zeros((ROWS, COLS), bool); filled[cy[idx][isS[idx]], cx[idx][isS[idx]]] = True
        letters = int(filled.sum())
    else:
        letters = 0

    # star burst at the tip
    if t > DRAW[1] - 0.3:
        tp = rot_x(rot_y(TIP[None], ang), tilt)
        x, y, _ = project(tp)
        tx, ty = int(x[0] / CW), int(y[0] / CH)
        pulse = 0.5 + 0.5 * np.sin((t - DRAW[1]) * 2.2)
        pulse = pulse ** 2 * smooth((t - DRAW[1] + 0.3) / 0.5)
        if t < DRAW[1] + 0.8:
            pulse = max(pulse, smooth((t - DRAW[1] + 0.3) / 0.3) * (1 - smooth((t - DRAW[1] - 0.3) / 0.5)))
        Lv, Lh = int(3 + 7 * pulse), int(4 + 12 * pulse)
        for k in range(-Lv, Lv + 1):
            yy = ty + k
            if 0 <= yy < ROWS and 0 <= tx < COLS and k != 0:
                G[yy, tx] = CI["|"]; C[yy, tx] = LIME * pulse * (1 - abs(k) / (Lv + 1))
        for k in range(-Lh, Lh + 1):
            xx = tx + k
            if 0 <= xx < COLS and 0 <= ty < ROWS and k != 0:
                G[ty, xx] = CI["-"]; C[ty, xx] = LIME * pulse * (1 - abs(k) / (Lh + 1))
        if 0 <= ty < ROWS and 0 <= tx < COLS and pulse > 0.05:
            G[ty, tx] = CI["*"]; C[ty, tx] = np.array([245, 255, 210]) * min(1, 0.4 + pulse)

    tiles = atlas[G][..., None] * C[:, :, None, None, :]
    img = tiles.transpose(0, 2, 1, 3, 4).reshape(ROWS * CH, COLS * CW, 3)

    # floor reflection
    floor = int(CY + F * 1.0 / CAM + 40)
    hgt = 150
    src = img[floor - hgt:floor][::-1].copy()
    fade = np.linspace(0.22, 0.0, hgt)[:, None, None]
    img[floor:floor + hgt] = np.maximum(img[floor:floor + hgt], src * fade)

    frame_img = np.zeros((H, W, 3)) + BG
    frame_img[RY0:RY0 + ROWS * CH, :COLS * CW] = np.maximum(frame_img[RY0:RY0 + ROWS * CH, :COLS * CW], img)
    # code panel bg
    frame_img[1150:] = np.array([9, 15, 11])
    out = Image.fromarray(np.clip(frame_img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(out)
    # HUD
    d.text((W - 24, 70), f"t {t:05.2f}s", font=sfont, fill=(198, 244, 50), anchor="ra")
    d.text((24, 70), SITE, font=sfont, fill=(110, 136, 122))
    hy = 1068
    d.text((24, hy), HANDLE, font=sfont, fill=(130, 156, 142))
    d.text((24, hy + 22), f"rotation {np.degrees(ang) % 360:06.2f}°   glints {n_glint}", font=sfont, fill=(100, 124, 112))
    d.text((24, hy + 44), f"letters {letters:,}", font=sfont, fill=(100, 124, 112))
    d.line([0, 1150, W, 1150], fill=(30, 44, 36), width=1)
    d.text((14, 1158), SCRIPT, font=sfont, fill=(110, 136, 122))
    d.text((W - 14, 1158), "running · 30 fps", font=sfont, fill=(110, 136, 122), anchor="ra")
    draw_code(d, t)

    a = np.asarray(out).astype(float)
    fo = smooth((DUR - t) / 0.6)
    fi = smooth(t / 0.25)
    a = BG + (a - BG) * min(fo, fi)
    return a.astype(np.uint8)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        for tt in [3.5, 9.0]:
            Image.fromarray(frame(tt)).save(os.path.join(HERE, f"preview_{tt}s.png"))
        print("saved preview images"); sys.exit()
    import imageio_ffmpeg
    out = os.path.join(HERE, OUTPUT)
    total = int(DUR * FPS)
    p = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "17",
                          "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    for i in range(total):
        p.stdin.write(frame(i / FPS).tobytes())
        if i % 30 == 0: print(f"rendering {i * 100 // total}%", end="\r", flush=True)
    p.stdin.close(); p.wait()
    print("done ->", out)
